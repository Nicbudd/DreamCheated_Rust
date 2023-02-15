## Benchmarks

### Environment

All benchmarks run on following device:

```
CPU: AMD Ryzen 7 2700X @ 3.7GHz
RAM: 2x8GB DDR4 (A-DATA) 3000 MT/s
OS: Pop!_OS 22.04 LTS
Kernel: 6.0.12-76060006-generic
```
Care was taken to attempt to remove all other unnessecary CPU processes.

Tests were initialized with the following string in dream.txt:
```
209,34,0,1970-01-01 00:00:00,0,0,0
```
Modified from the results from a long duration simulation. This is to remove any initiailization slowdowns some algorithms experience.

Tests run for approximately 1 minute. Tests run on 2023-02-14 unless specified otherwise.

### Results

|Language|Implementation|Attempts/Second|Parallel?|Notes|
|--------|--------------|---------------|---------|-----|
|Rust|rustc 1.65.0, --release|975.588 MA/s|Yes|**Thermal throttled during testing**<br>CPU temp to 83C, CPU clocks dipped from 3.9GHz to 3.7GHz, speed dropped regularly from peak of 1008.62 KA/s @&nbsp;t&nbsp;=&nbsp;30&nbsp;sec.|
|Rust|rustc 1.65.0, debug mode|52.920 MA/s|Yes|**Thermal throttled during testing**<br>CPU temp to 84C, CPU clocks dipped from 3.9GHz to 3.7GHz, speed dropped regularly from peak of 54.214 KA/s @&nbsp;t&nbsp;=&nbsp;20&nbsp;sec.|
|Java|openjdk 11.0.17,<br>Ubuntu Jammy Repos|16.509 MA/s|No|Does not have ability to start from file, was ran starting uninitialized.|
|C|gcc 11.3.0, `-O3`<br>Ubuntu Jammy Repos|223.567 KA/s|No||
|C|clang 14.0.0, `-O3`<br>Ubuntu Jammy Repos|222.702 KA/s|No||
|Python|PyPy 3.8.13 [GCC 11.2.0],<br>Ubuntu Jammy Repos|139.162 KA/s|No||
|Python|Python 3.10.6 [GCC 11.3.0],<br>Ubuntu Jammy Repos|33.398 KA/s|No||


