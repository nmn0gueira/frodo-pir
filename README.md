# SPIRro

This repository contains an implementation of SPIRro, a Symmetric PIR scheme derived from FrodoPIR, utilizing noise flooding and hash-based constructions to improve database privacy.

## Requirements

The source code can be built, tested, and benchmarked using [Docker](#using-docker).

In order to [natively](#native) build, run, test and benchmark the library, you will need the following:

```
  Rust >= 1.61.0
  Cargo
  Make
  Python3 >= 3.9.7
```

To obtain our performance numbers as reported in our paper, we run our benchmarks on an AWS EC2 c5n.metal instance.

## Quickstart

### Local

#### Building

To install the latest version of Rust, use the following command (you can also check how to install on the [Rust documentation](https://www.rust-lang.org/tools/install)):

```
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

To build the library, run:

```
  make build
```

#### Testing

To run the tests:

```
  make test
```

We test:

* A client and server workflow when using SPIRro (10 times).
* A test to check that the library fails if parameters are reused.

If all test build and run correctly, you should see an `ok` next to them.

#### Documentation

To view documentation (in a web browser manner):

```
  make docs
```

#### Benchmarking

To run a specific set of benchmarks, run (note that this process is slow):

```
  make bench
```

This command will execute client query benchmarks and Database generation benchmarks (for more details, see the `benches/bench.rs` file).

To run all benchmarks (note that this process is very slow, it takes around 30 minutes):

```
  make bench-all
```

This command will execute client query benchmarks and Database generation benchmarks for 16, 17, 18, 19 and 20 Number of DB items (log(m)). The results of these benchmarks can be found on Table 6 of our paper.

In order to see the results of the benchmarks, navigate to the `benchmarks-x.txt` file.

### Using Docker

The same notes as in the previous section also apply here.

Build Docker image:

```
docker build -t spirro .
```

Build and run tests:

```
docker run --rm spirro
```

Run Docker image interactively (from here, you can run any of the `make` commands below):

```
docker run --rm -it --entrypoint /bin/bash spirro
```
