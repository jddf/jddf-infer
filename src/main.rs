use chrono::DateTime;
use clap::{App, AppSettings, Arg};
use failure::{format_err, Error};
use jddf::{Form, Schema, Type};
use json_pointer::JsonPointer;
use serde_json::Value;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::fs::File;
use std::io::stdin;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Read;

fn main() -> Result<(), Error> {
    let matches = App::new("jddf-infer")
        .version("0.1.0")
        .about("Infers a JDDF schema from lines of JSON")
        .setting(AppSettings::ColoredHelp)
        .arg(
            Arg::with_name("INPUT")
                .help("Where to read examples from. Dash (hypen) indicates stdin")
                .default_value("-")
        )
        .arg(
            Arg::with_name("values-hint")
                .help("Advise the inferrer that the given path points to a values form. If this hint is proven wrong, a properties form will be emitted instead. This flag can be provided multiple times.")
                .multiple(true)
                .number_of_values(1)
                .long("values-hint"),
        )
        .arg(
            Arg::with_name("discriminator-hint")
                .help("Advise the inferrer that the given path points to a discriminator tag. If this hint is proven wrong, an empty form will be emitted instead. This flag can be provided multiple times.")
                .multiple(true)
                .number_of_values(1)
                .long("discriminator-hint"),
        )
        .get_matches();

    let reader = BufReader::new(match matches.value_of("INPUT").unwrap() {
        "-" => Box::new(stdin()) as Box<Read>,
        file @ _ => Box::new(File::open(file)?) as Box<Read>,
    });

    let mut hints = InferHints::new();
    for hint in matches.values_of("values-hint").unwrap_or_default() {
        let mut ptr: JsonPointer<String, Vec<String>> = hint
            .parse()
            .map_err(|e| format_err!("cannot parse as JSON Pointer: {:?}", e))?;

        // A bit of a hack: the json-pointer crate does not support just getting
        // the components. So we instead pop them out and put them into a deque.
        let mut path = VecDeque::new();
        while let Some(tok) = ptr.pop() {
            path.push_front(tok);
        }

        hints.add_values_hint(path.as_slices().0);
    }

    for hint in matches.values_of("discriminator-hint").unwrap_or_default() {
        let mut ptr: JsonPointer<String, Vec<String>> = hint
            .parse()
            .map_err(|e| format_err!("cannot parse as JSON Pointer: {:?}", e))?;

        // See note above regarding why we are doing this popping.
        let mut path = VecDeque::new();
        while let Some(tok) = ptr.pop() {
            path.push_front(tok);
        }

        let path = path.as_slices().0;
        let (tag, discriminator_path) = path.split_last().unwrap();

        hints.add_discriminator_hint(discriminator_path, tag.clone());
    }

    let mut inference = InferredSchema::Unknown;
    for line in reader.lines() {
        inference = inference.infer(serde_json::from_str(&line?)?, Some(&hints));
    }

    let serde_schema = inference.into_schema().into_serde();
    println!("{}", serde_json::to_string(&serde_schema)?);

    Ok(())
}

#[derive(Debug)]
struct InferHints {
    values: bool,
    discriminator_tag: Option<String>,
    children: HashMap<String, InferHints>,
}

impl InferHints {
    fn new() -> InferHints {
        InferHints {
            values: false,
            discriminator_tag: None,
            children: HashMap::new(),
        }
    }

    fn add_values_hint(&mut self, path: &[String]) {
        if path.is_empty() {
            self.values = true;
        } else {
            self.children
                .entry(path[0].clone())
                .or_insert(InferHints::new())
                .add_values_hint(&path[1..]);
        }
    }

    fn add_discriminator_hint(&mut self, path: &[String], tag: String) {
        if path.is_empty() {
            self.discriminator_tag = Some(tag);
        } else {
            self.children
                .entry(path[0].clone())
                .or_insert(InferHints::new())
                .add_discriminator_hint(&path[1..], tag);
        }
    }
}

#[derive(Debug)]
enum InferredSchema {
    Unknown,
    Any,
    Bool,
    Uint8,
    Int8,
    Int16,
    Uint16,
    Uint32,
    Int32,
    Float64,
    Timestamp,
    String,
    Array(Box<InferredSchema>),
    Properties(Box<InferredProperties>),
    Values(Box<InferredSchema>),
    Discriminator(String, HashMap<String, InferredSchema>),
}

#[derive(Debug)]
struct InferredProperties {
    required: HashMap<String, InferredSchema>,
    optional: HashMap<String, InferredSchema>,
}

impl InferredSchema {
    fn infer(self, value: Value, hints: Option<&InferHints>) -> InferredSchema {
        match (self, value) {
            (InferredSchema::Unknown, Value::Null) => InferredSchema::Any,
            (InferredSchema::Unknown, Value::Bool(_)) => InferredSchema::Bool,
            (InferredSchema::Unknown, Value::Number(n)) => minimum_number_type(n),
            (InferredSchema::Unknown, Value::String(s)) => {
                if DateTime::parse_from_rfc3339(&s).is_ok() {
                    InferredSchema::Timestamp
                } else {
                    InferredSchema::String
                }
            }
            (InferredSchema::Unknown, Value::Array(vals)) => {
                let mut sub_infer = InferredSchema::Unknown;
                for v in vals {
                    sub_infer = sub_infer.infer(v, hints.and_then(|h| h.children.get("-")));
                }

                InferredSchema::Array(Box::new(sub_infer))
            }
            (InferredSchema::Unknown, Value::Object(mut map)) => {
                if let Some(ref hint) = hints {
                    if hint.values {
                        let mut sub_infer = InferredSchema::Unknown;
                        for (k, v) in map {
                            sub_infer = InferredSchema::Unknown
                                .infer(v, hints.and_then(|h| h.children.get(&k)));
                        }

                        return InferredSchema::Values(Box::new(sub_infer));
                    } else if let Some(ref tag) = hint.discriminator_tag {
                        if let Some(Value::String(mapping_key)) = map.remove(tag) {
                            let infer_rest =
                                InferredSchema::Unknown.infer(Value::Object(map), hints);

                            let mut mapping = HashMap::new();
                            mapping.insert(mapping_key.to_owned(), infer_rest);

                            return InferredSchema::Discriminator(tag.to_owned(), mapping);
                        }
                    }
                }

                let mut props = HashMap::new();
                for (k, v) in map {
                    let sub_infer =
                        InferredSchema::Unknown.infer(v, hints.and_then(|h| h.children.get(&k)));
                    props.insert(k, sub_infer);
                }

                InferredSchema::Properties(Box::new(InferredProperties {
                    required: props,
                    optional: HashMap::new(),
                }))
            }
            (InferredSchema::Any, _) => InferredSchema::Any,
            (InferredSchema::Bool, Value::Bool(_)) => InferredSchema::Bool,
            (InferredSchema::Bool, _) => InferredSchema::Any,
            (InferredSchema::Uint8, Value::Number(n)) => match minimum_number_type(n) {
                InferredSchema::Uint8 => InferredSchema::Uint8,
                _ => InferredSchema::Float64,
            },
            (InferredSchema::Int8, Value::Number(n)) => match minimum_number_type(n) {
                InferredSchema::Int8 => InferredSchema::Int8,
                _ => InferredSchema::Float64,
            },
            (InferredSchema::Uint16, Value::Number(n)) => match minimum_number_type(n) {
                InferredSchema::Uint8 | InferredSchema::Int8 | InferredSchema::Uint16 => {
                    InferredSchema::Uint16
                }
                _ => InferredSchema::Float64,
            },
            (InferredSchema::Int16, Value::Number(n)) => match minimum_number_type(n) {
                InferredSchema::Uint8 | InferredSchema::Int8 | InferredSchema::Int16 => {
                    InferredSchema::Int16
                }
                _ => InferredSchema::Float64,
            },
            (InferredSchema::Uint32, Value::Number(n)) => match minimum_number_type(n) {
                InferredSchema::Uint8
                | InferredSchema::Int8
                | InferredSchema::Int16
                | InferredSchema::Uint16
                | InferredSchema::Uint32 => InferredSchema::Uint32,
                _ => InferredSchema::Float64,
            },
            (InferredSchema::Int32, Value::Number(n)) => match minimum_number_type(n) {
                InferredSchema::Uint8
                | InferredSchema::Int8
                | InferredSchema::Int16
                | InferredSchema::Uint16
                | InferredSchema::Int32 => InferredSchema::Int32,
                _ => InferredSchema::Float64,
            },
            (InferredSchema::Float64, Value::Number(_)) => InferredSchema::Float64,
            (InferredSchema::Uint8, _)
            | (InferredSchema::Int8, _)
            | (InferredSchema::Uint16, _)
            | (InferredSchema::Int16, _)
            | (InferredSchema::Uint32, _)
            | (InferredSchema::Int32, _)
            | (InferredSchema::Float64, _) => InferredSchema::Any,
            (InferredSchema::Timestamp, Value::String(s)) => {
                if DateTime::parse_from_rfc3339(&s).is_ok() {
                    InferredSchema::Timestamp
                } else {
                    InferredSchema::String
                }
            }
            (InferredSchema::Timestamp, _) => InferredSchema::Any,
            (InferredSchema::String, Value::String(_)) => InferredSchema::String,
            (InferredSchema::String, _) => InferredSchema::Any,
            (InferredSchema::Array(prior), Value::Array(vals)) => {
                let mut sub_infer = *prior;
                for v in vals {
                    sub_infer = sub_infer.infer(v, hints.and_then(|h| h.children.get("-")));
                }

                InferredSchema::Array(Box::new(sub_infer))
            }
            (InferredSchema::Array(_), _) => InferredSchema::Any,
            (InferredSchema::Properties(mut prior), Value::Object(map)) => {
                let missing_required_keys: Vec<_> = prior
                    .required
                    .keys()
                    .filter(|k| !map.contains_key(k.clone()))
                    .cloned()
                    .collect();
                for k in missing_required_keys {
                    let sub_infer = prior.required.remove(&k).unwrap();
                    prior.optional.insert(k, sub_infer);
                }

                for (k, v) in map {
                    if prior.required.contains_key(&k) {
                        let sub_infer = prior
                            .required
                            .remove(&k)
                            .unwrap()
                            .infer(v, hints.and_then(|h| h.children.get(&k)));
                        prior.required.insert(k, sub_infer);
                    } else if prior.optional.contains_key(&k) {
                        let sub_infer = prior
                            .optional
                            .remove(&k)
                            .unwrap()
                            .infer(v, hints.and_then(|h| h.children.get(&k)));
                        prior.optional.insert(k, sub_infer);
                    } else {
                        let sub_infer = InferredSchema::Unknown
                            .infer(v, hints.and_then(|h| h.children.get(&k)));
                        prior.optional.insert(k, sub_infer);
                    }
                }

                InferredSchema::Properties(prior)
            }
            (InferredSchema::Properties(_), _) => InferredSchema::Any,
            (InferredSchema::Values(prior), Value::Object(map)) => {
                let mut sub_infer = *prior;
                for (k, v) in map {
                    sub_infer =
                        InferredSchema::Unknown.infer(v, hints.and_then(|h| h.children.get(&k)));
                }

                return InferredSchema::Values(Box::new(sub_infer));
            }
            (InferredSchema::Values(_), _) => InferredSchema::Any,
            (InferredSchema::Discriminator(tag, mut mapping), Value::Object(mut map)) => {
                let mapping_key = map.remove(&tag);
                if let Some(Value::String(mapping_key_str)) = mapping_key {
                    if !mapping.contains_key(&mapping_key_str) {
                        mapping.insert(mapping_key_str.clone(), InferredSchema::Unknown);
                    }

                    let sub_infer = mapping
                        .remove(&mapping_key_str)
                        .unwrap()
                        .infer(Value::Object(map), hints);
                    mapping.insert(mapping_key_str, sub_infer);

                    InferredSchema::Discriminator(tag, mapping)
                } else {
                    // The hint was wrong. Retroactively re-computing a
                    // properties form is quite complex, and ultimately the user
                    // is likely going to be disappointed either way.
                    //
                    // So to keep this error condition simple, we simply
                    // downgrade to "any".
                    InferredSchema::Any
                }
            }
            (InferredSchema::Discriminator(_, _), _) => InferredSchema::Any,
        }
    }

    fn into_schema(self) -> Schema {
        let form = match self {
            InferredSchema::Unknown => Form::Empty,
            InferredSchema::Any => Form::Empty,
            InferredSchema::Bool => Form::Type(Type::Boolean),
            InferredSchema::Uint8 => Form::Type(Type::Uint8),
            InferredSchema::Int8 => Form::Type(Type::Int8),
            InferredSchema::Uint16 => Form::Type(Type::Uint16),
            InferredSchema::Int16 => Form::Type(Type::Int16),
            InferredSchema::Uint32 => Form::Type(Type::Uint32),
            InferredSchema::Int32 => Form::Type(Type::Int32),
            InferredSchema::Float64 => Form::Type(Type::Float64),
            InferredSchema::String => Form::Type(Type::String),
            InferredSchema::Timestamp => Form::Type(Type::Timestamp),
            InferredSchema::Array(sub_infer) => Form::Elements(sub_infer.into_schema()),
            InferredSchema::Properties(props) => {
                let has_required = !props.required.is_empty();

                Form::Properties {
                    required: props
                        .required
                        .into_iter()
                        .map(|(k, v)| (k, v.into_schema()))
                        .collect(),
                    optional: props
                        .optional
                        .into_iter()
                        .map(|(k, v)| (k, v.into_schema()))
                        .collect(),
                    has_required,
                    allow_additional: false,
                }
            }
            InferredSchema::Values(sub_infer) => Form::Values(sub_infer.into_schema()),
            InferredSchema::Discriminator(tag, mapping) => Form::Discriminator(
                tag,
                mapping
                    .into_iter()
                    .map(|(k, v)| (k, v.into_schema()))
                    .collect(),
            ),
        };

        Schema::from_parts(None, Box::new(form), HashMap::new())
    }
}

fn minimum_number_type(n: serde_json::Number) -> InferredSchema {
    let n = n.as_f64().unwrap();
    if n >= 0.0 && n <= 255.0 {
        InferredSchema::Uint8
    } else if n >= -128.0 && n <= 127.0 {
        InferredSchema::Int8
    } else if n >= 0.0 && n <= 65535.0 {
        InferredSchema::Uint16
    } else if n >= -32768.0 && n <= 32767.0 {
        InferredSchema::Int16
    } else if n >= 0.0 && n <= 4294967295.0 {
        InferredSchema::Uint32
    } else if n >= -2147483648.0 && n <= 2147483647.0 {
        InferredSchema::Int32
    } else {
        InferredSchema::Float64
    }
}
