#include <data_transformer/data_transformer.hpp>

#include <ctime>
#include <algorithm>
#include <iterator>
#include <random>
#include <memory>

#include <boost/regex.hpp>
#include <boost/range/iterator_range.hpp>

#include <opencv2/hdf.hpp>
//#include <opencv2/dnn.hpp>

namespace DataTransformer
{

  const rgbImageDepthPairList& getImageDepthPairFilenames( const std::string& _pathToImages )
  {
    boost::filesystem::path datasetDir( _pathToImages );

    rgbImageDepthPairList* rPairVector = new rgbImageDepthPairList;
    
    if( boost::filesystem::is_directory( datasetDir ) )
    { 
      uint32_t totalFiles = 0;
      uint32_t totalPairs = 0;
      const boost::regex filter( "\\S*rgb\\S*" );
      
      for( const auto& dirIt : boost::make_iterator_range( 
                                      boost::filesystem::recursive_directory_iterator( datasetDir )
                                      , {} ) )
      {
          
        if( boost::filesystem::is_regular_file( dirIt ) )
        {
          totalFiles++;
          boost::smatch what;
          
          if(  boost::regex_match( dirIt.path().filename().string(), what, filter )
            && dirIt.path().extension() == ".png" )
          {
            std::string depthFilename        = boost::regex_replace( dirIt.path().filename().string()
                                                                  , boost::regex( "rgb" )
                                                                  , "depth" );
            boost::filesystem::path depthDir( dirIt.path().parent_path().string() + "/" + depthFilename );
            
            if( boost::filesystem::exists( depthDir ) )
            {
              rgbImageDepthPair currentRgbDepthPair;
              
              currentRgbDepthPair.first  = dirIt.path();
              currentRgbDepthPair.second = depthDir;
              
              rPairVector->push_back( rgbImageDepthPair( dirIt.path(), depthDir ) );
              
              totalPairs++;
            }//if( boost::filesystem::exists ... )
            
          }//if( boost::regex_match ... )
          
        }//if( boost::filesystem::is_regular_file ... )
        
        
      }//for( const auto& dirIt : ... )
      
    }
    
    return *rPairVector;
    
  }//const rgbImageDepthPairList& getImageDepthPairFilenames( const std::string& _pathToImages )

  void shuffleData( rgbImageDepthPairList& _pairList )
  {
    std::random_shuffle( _pairList.begin(), _pairList.end() );
  }//void shuffleData(...)

  
  void cropData ( cv::Mat&       _image
                , cv::Mat&       _depth
                , const uint32_t _cropWidth
                , const uint32_t _cropHeight
                , const uint8_t  _cropType )
  {
    if( _cropType == CROP_NONE )
    {
      return;
    }
    
    if(  _image.size().width  != _depth.size().width 
      || _image.size().height != _depth.size().height )
    {
      throw cv::Exception();
    }
    
    cv::Rect cropArea;
    if( _cropType == CROP_RANDOM )
    {
      srand( time(NULL) );
      
      cropArea.x      = rand() % ( _image.size().width  - _cropWidth );
      cropArea.y      = rand() % ( _image.size().height - _cropHeight );
    }
    if( _cropType == CROP_CENTER )
    {
      cropArea.x      = _image.size().width  / 2.0 - _cropWidth  / 2.0;
      cropArea.y      = _image.size().height / 2.0 - _cropHeight / 2.0;
    }
    cropArea.width  = _cropWidth;
    cropArea.height = _cropHeight;
    
    cv::Mat croppedImage = _image( cropArea );
    croppedImage.copyTo( _image );
    
    cv::Mat croppedDepth = _depth( cropArea );
    croppedDepth.copyTo( _depth );
  }//void cropData()
  
  meanDev_t computeDataMeanDev( const rgbImageDepthPairList& _pairList, const uint8_t _verbose )
  {
    bool isMeanDevInit = false;
    
    meanDev_t rMeanDev;
    
    //cv::accumulate???
    
    for( const auto& it : _pairList )
    {
      cv::Mat image = cv::imread( it.first.c_str(), cv::IMREAD_COLOR );
      cv::Mat depth = cv::imread( it.second.c_str(), cv::IMREAD_GRAYSCALE );
      
      cv::Mat imageMean;
      cv::Mat imageDev;
      cv::Mat depthMean;
      cv::Mat depthDev;
      
      cv::meanStdDev( image, imageMean, imageDev );
      cv::meanStdDev( depth, depthMean, depthDev );
      
      if( !isMeanDevInit )
      {
        imageMean.copyTo( rMeanDev.imageMean );
        imageDev.copyTo( rMeanDev.imageDev );
        depthMean.copyTo( rMeanDev.depthMean );
        depthDev.copyTo( rMeanDev.depthDev );
        
        isMeanDevInit = true;
      }
      else
      {
        rMeanDev.imageMean += imageMean;
        rMeanDev.imageDev  += imageDev;
        rMeanDev.depthMean += depthMean;
        rMeanDev.depthDev  += depthDev;
      }
      
      if( _verbose )
      {
        std::cout << "[computeDataMeanDev]: " << std::distance( _pairList.begin(), std::find( _pairList.begin(), _pairList.end(), it ) ) 
                  << "/" << _pairList.size() << std::endl;
      }
    }
    rMeanDev.imageMean /= _pairList.size();
    rMeanDev.imageDev  /= _pairList.size();
    rMeanDev.depthMean /= _pairList.size();
    rMeanDev.depthDev  /= _pairList.size();
    
    if( _verbose )
    {
      std::cout << "[computeDataMeanDev] Final image mean: " << std::endl
                << rMeanDev.imageMean << std::endl
                << "[computeDataMeanDev] Final image dev: " << std::endl
                << rMeanDev.imageDev << std::endl
                << "[computeDataMeanDev] Final depth mean: " << std::endl
                << rMeanDev.depthMean << std::endl
                << "[computeDataMeanDev] Final depth dev: " << std::endl
                << rMeanDev.depthDev << std::endl;
    } 
    
    return rMeanDev;
  }//meanDev_t computeDataMeanDev( const rgbImageDepthPairList& _pairList, const uint8_t _verbose )
  
  void applyMeanDev(cv::Mat& _image, cv::Mat& _depth, const meanDev_t _meanDev)
  {
    cv::Scalar scalarImageMean;
    cv::Scalar scalarImageDev;
    cv::Scalar scalarDepthMean;
    cv::Scalar scalarDepthDev;
    
    for( int i = 0; i < _image.channels(); ++i )
    {
      scalarImageMean[i] = _meanDev.imageMean.at<double>( 0, i );
      scalarImageDev[i]  = _meanDev.imageDev.at<double>( 0, i );
    }
    
    for( int i = 0; i < _depth.channels(); ++i )
    {
      scalarDepthMean[i] = _meanDev.depthMean.at<double>( 0, i );
      scalarDepthDev[i]  = _meanDev.depthDev.at<double>( 0, i );
    }

    cv::subtract( _image, scalarImageMean, _image );
    cv::divide( _image, scalarImageMean, _image );
    cv::subtract( _depth, scalarDepthMean, _depth );
    cv::divide( _depth, scalarDepthDev, _depth );
  }//void applyMeanDev(cv::Mat _image, cv::Mat _depth, const meanDev_t _meanDev)

  void composeHdf5Dataset( const rgbImageDepthPairList& _pairList
                         , const std::string& _datasetFilename
                         , const bool     _applyMeanDev
                         , const uint8_t  _cropType  
                         , const uint32_t _cropWidth 
                         , const uint32_t _cropHeight )
  {
    std::unique_ptr<int[]> offset;
    //std::make_unique<>
    
    cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open( _datasetFilename.c_str() );
    meanDev_t              dsMeanDev;
    
    if( _applyMeanDev )
    {
      dsMeanDev = computeDataMeanDev( _pairList, 0 );
    }
    
    for( const auto& it : _pairList )
    {
      cv::Mat image = cv::imread( it.first.c_str(), cv::IMREAD_COLOR );
      cv::Mat depth = cv::imread( it.second.c_str(), cv::IMREAD_GRAYSCALE );
      
      if( image.channels() == 3 )
      {
        image.convertTo( image, CV_32FC3 );
      }
      else
      {
        image.convertTo( image, CV_32FC1 );
      }
      
      if( _cropType != CROP_NONE )
      {
        //CV_Assert( _cropWidth <= 0 || _cropHeight <= 0 || _cropWidth > image.cols || _cropHeight > image.rows  );
        cropData( image, depth, _cropWidth, _cropHeight, _cropType );
      }
      
      depth.convertTo( depth, CV_32FC1 );
  
      if( _applyMeanDev )
      {
        applyMeanDev( image, depth, dsMeanDev );
      }
      
      cv::Mat imageBlob = blobFromImage( image );
      cv::Mat depthBlob = blobFromImage( depth );
      
      std::cout << image.rows << " " << image.cols << std::endl;
      std::cout << "Blob: " << imageBlob.rows << " " << imageBlob.cols << std::endl;
      std::cout << "Blob: " << imageBlob.size[0] << " " << imageBlob.size[1] << " " << imageBlob.size[2] << " " << imageBlob.size[3] << std::endl;
      
      if( !offset )
      {
        offset = std::unique_ptr<int[]>( new int[ imageBlob.dims ] );
        for( int i = 0; i < imageBlob.dims; ++i )
        {
          offset[i] = 0;
        }
      }
      
      if( !h5io->hlexists( "data" ) )
      {        
        std::unique_ptr<int[]> dsChunks( new int[ imageBlob.dims ] );
        std::unique_ptr<int[]> dsDims( new int[ imageBlob.dims ] );
        
        for( int i = 0; i < imageBlob.dims; ++i )
        {
          dsDims[i]   = cv::hdf::HDF5::H5_UNLIMITED;
          dsChunks[i] = imageBlob.size[i];
        }
        h5io->dscreate( imageBlob.dims, dsDims.get(), imageBlob.type(), "data", cv::hdf::HDF5::H5_NONE, dsChunks.get() );
      }
      
      if( !h5io->hlexists( "label" ) )
      {        
        std::unique_ptr<int[]> dsChunks( new int[ imageBlob.dims ] );
        std::unique_ptr<int[]> dsDims( new int[ imageBlob.dims ] );
        
        for( int i = 0; i < depthBlob.dims; ++i )
        {
          dsDims[i]   = cv::hdf::HDF5::H5_UNLIMITED;
          dsChunks[i] = imageBlob.size[i];
        }
        h5io->dscreate( depthBlob.dims, dsDims.get(), depthBlob.type(), "label", cv::hdf::HDF5::H5_NONE, dsChunks.get() );        
      }
      
      h5io->dsinsert( imageBlob, "data", offset.get() );
      h5io->dsinsert( depthBlob, "label", offset.get() );
      offset[0] += 1;
    }
    
    h5io->close();
  }//void composeHdf5Dataset( const rgbImageDepthPairList& _pairList ...

  void blobFromImages( cv::InputArrayOfArrays images_
                     , cv::OutputArray        blob_
                     , double                 scalefactor
                     , cv::Size               size
                     , const                  cv::Scalar& mean_
                     , bool                   swapRB
                     , bool                   crop
                     , int                    ddepth )
  {
    //CV_TRACE_FUNCTION();
    //CV_CheckType(ddepth, ddepth == CV_32F || ddepth == CV_8U, "Blob depth should be CV_32F or CV_8U");
    if (ddepth == CV_8U)
    {
        //cv::CV_CheckEQ(scalefactor, 1.0, "Scaling is not supported for CV_8U blob depth");
        CV_Assert(mean_ == cv::Scalar() && "Mean subtraction is not supported for CV_8U blob depth");
    }

    std::vector<cv::Mat> images;
    images_.getMatVector(images);
    CV_Assert(!images.empty());
    for (int i = 0; i < images.size(); i++)
    {
        cv::Size imgSize = images[i].size();
        if (size == cv::Size())
            size = imgSize;
        if (size != imgSize)
        {
            if(crop)
            {
              float resizeFactor = std::max(size.width / (float)imgSize.width,
                                            size.height / (float)imgSize.height);
              cv::resize(images[i], images[i], cv::Size(), resizeFactor, resizeFactor, cv::INTER_LINEAR);
              cv::Rect crop(cv::Point(0.5 * (images[i].cols - size.width),
                              0.5 * (images[i].rows - size.height)),
                        size);
              images[i] = images[i](crop);
            }
            else
              cv::resize(images[i], images[i], size, 0, 0, cv::INTER_LINEAR);
        }
        if(images[i].depth() == CV_8U && ddepth == CV_32F)
            images[i].convertTo(images[i], CV_32F);
        cv::Scalar mean = mean_;
        if (swapRB)
            std::swap(mean[0], mean[2]);

        images[i] -= mean;
        images[i] *= scalefactor;
    }

    size_t i, nimages = images.size();
    cv::Mat image0 = images[0];
    int nch = image0.channels();
    CV_Assert(image0.dims == 2);
    cv::Mat image;
    if (nch == 3 || nch == 4)
    {
        int sz[] = { (int)nimages, nch, image0.rows, image0.cols };
        blob_.create(4, sz, ddepth);
        cv::Mat blob = blob_.getMat();
        cv::Mat ch[4];

        for( i = 0; i < nimages; i++ )
        {
            image = images[i];
            CV_Assert(image.depth() == blob_.depth());
            nch = image.channels();
            CV_Assert(image.dims == 2 && (nch == 3 || nch == 4));
            CV_Assert(image.size() == image0.size());

            for( int j = 0; j < nch; j++ )
                ch[j] = cv::Mat(image.rows, image.cols, ddepth, blob.ptr((int)i, j));
            if(swapRB)
                std::swap(ch[0], ch[2]);
            cv::split(image, ch);
        }
    }
    else
    {
       CV_Assert(nch == 1);
       int sz[] = { (int)nimages, 1, image0.rows, image0.cols };
       blob_.create(4, sz, ddepth);
       cv::Mat blob = blob_.getMat();

       for( i = 0; i < nimages; i++ )
       {
           cv::Mat image = images[i];
           CV_Assert(image.depth() == blob_.depth());
           nch = image.channels();
           CV_Assert(image.dims == 2 && (nch == 1));
           CV_Assert(image.size() == image0.size());

           image.copyTo(cv::Mat(image.rows, image.cols, ddepth, blob.ptr((int)i, 0)));
       }
    }
  }
  
  cv::Mat blobFromImage(cv::InputArray image, double scalefactor, const cv::Size& size,
                  const cv::Scalar& mean, bool swapRB, bool crop, int ddepth)
  {
    //cv::CV_TRACE_FUNCTION();
    cv::Mat blob;
    std::vector<cv::Mat> images(1, image.getMat());
    blobFromImages(images, blob, scalefactor, size, mean, swapRB, crop, ddepth);
    return blob;
  }
  
  
  //TODO: Release
  const testTrainSplit_t& splitTrainVal( const rgbImageDepthPair& _pairList
                                       , const uint8_t            _percent )
  {
    
  }
  
  
}// namespace DataTransformer














