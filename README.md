# Viola Jones Face Detector

## Usage:
```
$ python3 violajones.py --help
Usage: violajones.py [OPTIONS]

Options:
  -f, --faces TEXT       Path to directory containing face examples.
                         [required]
  -b, --background TEXT  Path to directory containing background examples.
                         [required]
  -l, --load TEXT        Load saved cascade configuration.
  -t, --test TEXT        Test image.
  -v, --verbose          Toggle for verbosity.
  --help                 Show this message and exit.

$ time python3 -OO violajones.py -f data/faces/ -b data/background/ -v > log.txt

$ time python3 -OO violajones.py -f data/faces/ -b data/background/ -l cascade_save.json -t data/class.jpg -v
Evaluating cascade in 207872 image patches.
After 1 cascade steps, 3601 potential faces.
After 2 cascade steps, 547 potential faces.
After 3 cascade steps, 116 potential faces.
After 4 cascade steps, 93 potential faces.
After 5 cascade steps, 93 potential faces.

real	0m11.274s
user	0m7.328s
sys	0m4.508s

```

