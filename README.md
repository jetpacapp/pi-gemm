Pi-GEMM
=======

This is a GPU-accelerated implementation of the GEMM matrix multiply function for the Raspberry Pi.

The core is an assembler loop for Broadcoms QPU processor, and is run as a custom program on their GPU.
It produces a substantial speedup compared to an optimized CPU version, with the included test running in 500ms, rather than 8,000 ms using the official Atlas library on Raspbian.

## Getting Started

Download the repo, run make, and then run `sudo ./gemm`.

If you don't already have Atlas installed for comparison, you can try running `sudo apt-get install libatlas-dev` on Raspbian.

## License

All code is under the BSD three-clause license, included in this folder as LICENSE.

The ccv2010.ntwk and ccv2012.ntwk network models were converted from files created as part of the [LibCCV project](http://libccv.org/) and are licensed under the Creative Commons Attribution 4.0 International License. To view a copy of this license, visit [http://creativecommons.org/licenses/by/4.0/](http://creativecommons.org/licenses/by/4.0/).

## Credits

Written by [Pete Warden](https://twitter.com/petewarden) at Jetpac Inc.
