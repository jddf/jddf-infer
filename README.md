# jddf-infer

`jddf-infer` generates a JDDF schema from example values. It's useful for when
you want to put together the skeleton of a schema quickly, just using example
data.

## Usage

See `jddf-infer --help` for detailed instructions, but essentially you run it
like this:

```text
$ echo '{"name": "John", "born": "1971-10-19T17:26:48+00:00"}' | jddf-infer
{"properties":{"born":{"type":"timestamp"},"name":{"type":"string"}}}
```

By default, `jddf-infer` reads example values from STDIN, and then produces an
inferred schema to STDOUT. To use a file instead of STDIN, you can call
`jddf-infer` as:

```text
jddf-infer examples.jsonl
```

`jddf-infer` requires that its input be a finite sequence of [JSON
Lines](jsonlines.org) -- one-line JSON values, separated by newlines.
Essentially, that means the input has to look something like:

```text
{ "a": "foo" }
123
"asdf"
[1, 2, 3]
```

### Providing Hints

In order to keep its output predictable and useful by default, `jddf-infer` will
never output the "values" or "discriminator" forms by default, as that would
require some heuristic to distinguish these forms from the more common
"properties" form.

To cirumvent this behavior, `jddf-infer` provides two flags, which can both be
passed as many times as required:

* `--values-hint=<json pointer>` takes a [JSON Pointer][rfc6901] which points to
  a path in the input which should be inferred to be of the "values" form.
* `--discriminator-hint=<json pointer>` takes a JSON Pointer which points to a
  path in the input which should be inferred to be the "tag" of a
  "discriminator" form.

In both cases, if the hint is proven to be wrong (for example, if it points to a
path which doesn't exist, or if example values contradict the hint), then the
hint will be ignored.

To pass a hint regarding the elements of an array, use `-` as the array index.

All of this is best explained with an example. By default, if you give
`jddf-infer` this input:

```json
{ "stuff": { "a": "foo", "b": "bar" }}
```

It will infer a "properties" form:

```text
$ echo '{ "stuff": { "a": "foo", "b": "bar" }}' | jddf-infer
{"properties":{"stuff":{"properties":{"a":{"type":"string"},"b":{"type":"string"}}}}}
```

But you can pass a `--values-hint` to force the object under `/stuff` to be
treated as a "values" form:

```text
$ echo '{ "stuff": { "a": "foo", "b": "bar" }}' | jddf-infer --values-hint=/stuff
{"properties":{"stuff":{"values":{"type":"string"}}}}
```

Similarly, imagine you had a file with these values:

```json
{ "stuff": { "arrayOf": "strings", "array": ["a", "b", "c"]}}
{ "stuff": { "arrayOf": "numbers", "array": [1, 2, 3]}}
{ "stuff": { "arrayOf": "booleans", "array": [true, false, true]}}
```

By default, `jddf-infer` will infer the following schema:

```text
$ cat examples.jsonl | jddf-infer
{"properties":{"stuff":{"properties":{"array":{"elements":{}},"arrayOf":{"type":"string"}}}}}
```

But if you want `jddf-infer` to treat `/stuff` as a discriminator form, where
`/stuff/arrayOf` is the discriminator tag, then provide a `--discriminator-hint`
pointing to `/stuff/arrayOf`:

```text
$ echo examples.jsonl | jddf-infer --discriminator-hint=/stuff/arrayOf | jq
{
  "properties": {
    "stuff": {
      "discriminator": {
        "tag": "arrayOf",
        "mapping": {
          "numbers": {
            "properties": {
              "array": {
                "elements": {
                  "type": "number"
                }
              }
            }
          },
          "strings": {
            "properties": {
              "array": {
                "elements": {
                  "type": "string"
                }
              }
            }
          },
          "booleans": {
            "properties": {
              "array": {
                "elements": {
                  "type": "boolean"
                }
              }
            }
          }
        }
      }
    }
  }
}
```

[rfc6901]: https://tools.ietf.org/html/rfc6901
