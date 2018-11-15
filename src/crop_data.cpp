#include <iostream>

#include <data_transformer/data_transformer.hpp>

int main( int argc, char** argv )
{
  boost::filesystem::path inPath( "../db" );
  boost::filesystem::path outPath( "../out" );
  
  uint32_t cropWidth = 320;
  uint32_t cropHeigh = 240;
  
  if( !boost::filesystem::exists( outPath ) )
  {
    boost::filesystem::create_directory( outPath );
  }
  
  boost::filesystem::path exePath = boost::filesystem::current_path();
  boost::filesystem::current_path( inPath );
  
  DataTransformer::rgbImageDepthPairList pairList = DataTransformer::getImageDepthPairFilenames( "." );
  
  //boost::
  
  for( const auto& it : pairList )
  {
    cv::Mat image = cv::imread( it.first.c_str(), cv::IMREAD_COLOR );
    cv::Mat depth = cv::imread( it.second.c_str(), cv::IMREAD_GRAYSCALE );
    
    boost::filesystem::current_path( outPath );
    
    std::string imageFilenameAndPath = it.first.string();
    imageFilenameAndPath.erase( 0, 1 );
    
    std::string depthFilenameAndPath = it.second.string();
    depthFilenameAndPath.erase( 0, 1 );
    
    boost::filesystem::path pathToOutImageFile = boost::filesystem::current_path() / imageFilenameAndPath;
    
    if( !boost::filesystem::exists( pathToOutImageFile.parent_path() ) )
    {
      boost::filesystem::create_directory( pathToOutImageFile.parent_path() );
    }
    
    DataTransformer::cropData( image, depth, cropWidth, cropHeigh, DataTransformer::CROP_RANDOM );
    
    cv::imwrite( ( boost::filesystem::current_path() / imageFilenameAndPath ).c_str(), image );
    cv::imwrite( ( boost::filesystem::current_path() / depthFilenameAndPath ).c_str(), depth );
    
    boost::filesystem::current_path( inPath );
  }
  
  
  
  
  return 0;
}
