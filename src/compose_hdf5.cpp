#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include <H5Cpp.h>

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/regex.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/hdf.hpp>

#include <data_transformer/data_transformer.hpp>

typedef std::pair< boost::filesystem::path, boost::filesystem::path > rgbDepthPair; 

int main( int argc, char** argv )
{
  const cv::String keys = 
    "{@in  i         |<none>     | path to directory, containing dataset                           }"
    "{@out o         |train.hdf5 | path to output hdf5 file                                        }"
    "{mean m         |false      | additional output                                               }"
    "{crop c         |0          | crop input image. 0 - crop none, 1 -crop random, 2 - center crop}"
    "{width w        |0          | crop width                                                      }"
    "{height h       |0          | crop height                                                     }"
    "{help h usage ? |           | print this message }";
  
  cv::CommandLineParser parser( argc, argv, keys );
  parser.about( "Utility to compose hdf5 file from the set of image-depth pairs" );
  
  if( parser.has("help") )
  {
    parser.printMessage();
    return 0;
  }
   
  std::string argPath = parser.get<cv::String>( "@in" );
  std::string argOut  = parser.get<cv::String>( "@out" );
  bool        argMean = parser.get<bool>( "mean" );
  int         argCrop = parser.get<int>( "crop" );
  int        argwidth = parser.get<int>( "width" );
  int       argheight = parser.get<int>( "height" );
  
  if (!parser.check())
  {
    parser.printErrors();
    return 0;
  }
  
  uint8_t cropType;
  
  switch( argCrop )
  {
    case 0 : cropType = DataTransformer::CROP_NONE; 
             break;
    case 1 : cropType = DataTransformer::CROP_RANDOM;
             break;
    case 2 : cropType = DataTransformer::CROP_CENTER;
             break;
    default: std::cout << "ERROR: Wrong crop argument. See help for usage" << std::endl;
             return 0;
  }
  
  boost::filesystem::path currentDir( argPath.c_str() );
  
  DataTransformer::rgbImageDepthPairList funcPairs; 
  funcPairs = DataTransformer::getImageDepthPairFilenames( currentDir.c_str() );
  
  std::cout << "Total images: " << funcPairs.size() << std::endl;
  
  DataTransformer::shuffleData( funcPairs );
  
  for( const auto& it : funcPairs )
  {
    std::cout << it.first << " " << it.second << std::endl;
  }
  
  
  DataTransformer::composeHdf5Dataset( funcPairs, argOut.c_str(), argMean, cropType, argwidth, argheight );
  
  
  return 0;
}
































