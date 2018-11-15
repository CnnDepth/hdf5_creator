# hdf5_creator
HDF5 database utils for Caffe 

## Compile
```bash
mkdir build && cd build
cmake ..
make
```

## Usage
```bash
./compose_hdf5 -h
Utility to compose hdf5 file from the set of image-depth pairs
Usage: compose_hdf5 [params] in out 

	-?, -h, --help, --usage (value:true)
		print this message
	-c, --crop (value:0)
		crop input image. 0 - crop none, 1 -crop random, 2 - center crop
	-h, --height (value:true)
		crop height
	-m, --mean (value:false)
		additional output
	-w, --width (value:0)
		crop width

	in (value:<none>)
		path to directory, containing dataset
	out (value:train.hdf5)
		path to output hdf5 file
```
