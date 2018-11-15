#ifndef DT_TRNSFRMR_HPP
#define DT_TRNSFRMR_HPP

#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core/base.hpp>
//#include <opencv2/core/check.hpp>

namespace DataTransformer
{
  typedef std::pair< boost::filesystem::path, boost::filesystem::path > rgbImageDepthPair; 
  typedef std::pair< rgbImageDepthPair, rgbImageDepthPair > testTrainSplit_t;
  typedef std::vector<rgbImageDepthPair> rgbImageDepthPairList;
  
  typedef struct 
  {
    cv::Mat imageMean;
    cv::Mat depthMean;
    cv::Mat imageDev;
    cv::Mat depthDev;
  }meanDev_t;
  
  /* If the new values are added, check cropData() function
   */
  const uint8_t CROP_NONE   = 0;
  const uint8_t CROP_RANDOM = 1;
  const uint8_t CROP_CENTER = 2;
  
  const rgbImageDepthPairList& getImageDepthPairFilenames( const std::string& _pathToImages );
  
  /*! \brief Shuffle the list of filename pairs
   *  \param _pairList List of filename pairs to shuffle
   */
  void shuffleData( rgbImageDepthPairList& _pairList );
  //const rgbImageDepthPairList& shuffleData( const rgbImageDepthPairList& _pairList );
  
  /*! \brief Function for cropping image-depth pairs.
   *  
   * This function takes 2 images as arguments and crops them the same way
   * so the data of image matches the data of depth. 
   * \param _image a cv::Mat RGB image
   * \param _depth a cv::Mat grayscale depth, that matches the _image. The sizes of both _image and _depth are expected to be the same
   * \param _cropWidth the size of the x-axis crop's area. Must be <= _image.rows()
   * \param _cropHeight the size of the y-axis crop's area. Must be <= _image.cols()
   */
  void cropData( cv::Mat&       _image
               , cv::Mat&       _depth
               , const uint32_t _cropWidth  = 0
               , const uint32_t _cropHeight = 0
               , const uint8_t  _cropType   = CROP_RANDOM );
  
  //void cropAndSaveData(  )
  
  //TODO: Change to cv::Scalar
  meanDev_t computeDataMeanDev( const rgbImageDepthPairList& _pairList, const uint8_t _verbose = 0 );
  
  void applyMeanDev( cv::Mat& _image, cv::Mat& _depth, const meanDev_t _meanDev );
  
  
  /* !TODO: Add description
   */
  const testTrainSplit_t& splitTrainVal( const rgbImageDepthPairList& _pairList
                                       , const uint8_t                _precent );
                                      
  /* !TODO: Add description
   * !TODO: Add split functionality
   */
  void composeHdf5Dataset( const rgbImageDepthPairList& _pairList
                         , const std::string&           _datasetFilename = "train.hdf5"
                         , const bool                   _appplyMeanDev = false
                         , const uint8_t                _cropType  = CROP_NONE
                         , const uint32_t               _cropWidth = 0
                         , const uint32_t               _cropHeight = 0 );
  /* Refer https://github.com/opencv/opencv/tree/master/modules/dnn
   *
   */
  void blobFromImages( cv::InputArrayOfArrays images_
                     , cv::OutputArray        blob_
                     , double                 scalefactor
                     , cv::Size               size
                     , const                  cv::Scalar& mean_
                     , bool                   swapRB
                     , bool                   crop
                     , int                    ddepth );
  
  
  /* Refer
   */
  cv::Mat blobFromImage(cv::InputArray image, double scalefactor = 1.0, const cv::Size& size = cv::Size(),
                  const cv::Scalar& mean = cv::Scalar(), bool swapRB = false, bool crop = false, int ddepth = CV_32F);
  
};


#endif //DT_TRNSFRMR_HPP
