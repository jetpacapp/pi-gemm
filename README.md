Pi-GEMM
=======

This is a GPU-accelerated implementation of the GEMM matrix multiply function for the Raspberry Pi.

The core is an assembler loop for Broadcoms QPU processor, and is run as a custom program on their GPU.
It produces a substantial speedup compared to an optimized CPU version, with the included test running in 500ms on my overclocked Pi, rather than 8,000 ms using the official Atlas library on Raspbian on the same device.

## Getting Started

Download the repo, run make, and then run `sudo ./gemm`.

If you don't already have Atlas installed for comparison, you can try running `sudo apt-get install libatlas-dev` on Raspbian.

## Notes

It always overwrites the output 'C' matrix, rather than incrementing it by 'beta'.

You have to run the program as 'su', so that the library can get direct access to the GPU.

## License

All code is under the BSD three-clause license, included in this folder as LICENSE.

## Credits

Written by [Pete Warden](https://twitter.com/petewarden) at Jetpac Inc.

Thanks to [eman](http://www.raspberrypi.org/forums/viewtopic.php?f=33&t=77231) on the Pi forums for the SHA-256 examples, [Andrew Holme](http://www.aholme.co.uk/) for creating the Fourier library, [Herman Hermitage](https://github.com/hermanhermitage/videocoreiv-qpu) for his QPU documentation work, and Broadcom for releasing the hardware specifications of their GPU! 