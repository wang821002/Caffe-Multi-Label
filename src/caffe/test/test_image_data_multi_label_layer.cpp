#ifdef USE_OPENCV
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/image_data_multi_label_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class ImageDataMultiLabelLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ImageDataMultiLabelLayerTest()
      : seed_(1701),
        blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
    Caffe::set_random_seed(seed_);
    // Create test input file.
    MakeTempFilename(&filename_);
    std::ofstream outfile(filename_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_;
    for (int i = 0; i < 5; ++i) {
      outfile << EXAMPLES_SOURCE_DIR "images/cat.jpg " << i << ' ' << i << std::endl;
    }
    outfile.close();
    // Create test input file for images of distinct sizes.
    MakeTempFilename(&filename_reshape_);
    std::ofstream reshapefile(filename_reshape_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_reshape_;
    reshapefile << EXAMPLES_SOURCE_DIR "images/cat.jpg " << 1 << ' ' << 0 << std::endl;
    reshapefile << EXAMPLES_SOURCE_DIR "images/fish-bike.jpg " << 0 << ' ' << 1
                << std::endl;
    reshapefile.close();
    // Create test input file for images with space in names
    MakeTempFilename(&filename_space_);
    std::ofstream spacefile(filename_space_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_space_;
    spacefile << EXAMPLES_SOURCE_DIR "images/cat.jpg " << 1 << ' ' << 0 << std::endl;
    spacefile << EXAMPLES_SOURCE_DIR "images/cat_gray.jpg " << 0 << ' ' << 1 << std::endl;
    spacefile.close();
  }

  virtual ~ImageDataMultiLabelLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
  }

  int seed_;
  string filename_;
  string filename_reshape_;
  string filename_space_;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ImageDataMultiLabelLayerTest, TestDtypesAndDevices);

TYPED_TEST(ImageDataMultiLabelLayerTest, TestRead) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataMultiLabelParameter* image_data_multi_label_param = param.mutable_image_data_multi_label_param();
  image_data_multi_label_param->set_batch_size(5);
  image_data_multi_label_param->set_source(this->filename_.c_str());
  std::cout << this->filename_.c_str() << std::endl;
  image_data_multi_label_param->set_shuffle(false);
  ImageDataMultiLabelLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 2);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i*2]);
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i*2+1]);
    }
  }
}

TYPED_TEST(ImageDataMultiLabelLayerTest, TestResize) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataMultiLabelParameter* image_data_multi_label_param = param.mutable_image_data_multi_label_param();
  image_data_multi_label_param->set_batch_size(5);
  image_data_multi_label_param->set_source(this->filename_.c_str());
  image_data_multi_label_param->set_new_height(256);
  image_data_multi_label_param->set_new_width(256);
  image_data_multi_label_param->set_shuffle(false);
  ImageDataMultiLabelLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 256);
  EXPECT_EQ(this->blob_top_data_->width(), 256);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 2);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
    	EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i*2]);
    	EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i*2+1]);
    }
  }
}

TYPED_TEST(ImageDataMultiLabelLayerTest, TestReshape) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataMultiLabelParameter* image_data_multi_label_param = param.mutable_image_data_multi_label_param();
  image_data_multi_label_param->set_batch_size(1);
  image_data_multi_label_param->set_source(this->filename_reshape_.c_str());
  image_data_multi_label_param->set_shuffle(false);
  ImageDataMultiLabelLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_label_->num(), 1);
  EXPECT_EQ(this->blob_top_label_->channels(), 2);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // cat.jpg
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 1);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  // fish-bike.jpg
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 1);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 323);
  EXPECT_EQ(this->blob_top_data_->width(), 481);
}

TYPED_TEST(ImageDataMultiLabelLayerTest, TestShuffle) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataMultiLabelParameter* image_data_multi_label_param = param.mutable_image_data_multi_label_param();
  image_data_multi_label_param->set_batch_size(5);
  image_data_multi_label_param->set_source(this->filename_.c_str());
  image_data_multi_label_param->set_shuffle(true);
  ImageDataMultiLabelLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 2);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
//  // Go through the data twice
//  for (int iter = 0; iter < 2; ++iter) {
//    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//    map<Dtype, int> values_to_indices;
//    int num_in_order = 0;
//    for (int i = 0; i < 5; ++i) {
//      Dtype value = this->blob_top_label_->cpu_data()[i*2];
//      // Check that the value has not been seen already (no duplicates).
//      EXPECT_EQ(values_to_indices.find(value), values_to_indices.end());
//      values_to_indices[value] = i;
//      num_in_order += (value == Dtype(i));
//    }
//    EXPECT_EQ(5, values_to_indices.size());
//    EXPECT_GT(5, num_in_order);
//  }
}

TYPED_TEST(ImageDataMultiLabelLayerTest, TestSpace) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  ImageDataMultiLabelParameter* image_data_multi_label_param = param.mutable_image_data_multi_label_param();
  image_data_multi_label_param->set_batch_size(1);
  image_data_multi_label_param->set_source(this->filename_space_.c_str());
  std::cout << this->filename_.c_str() << std::endl;
  image_data_multi_label_param->set_shuffle(false);
  ImageDataMultiLabelLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_label_->num(), 1);
  EXPECT_EQ(this->blob_top_label_->channels(), 2);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // cat.jpg
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 1);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  EXPECT_EQ(this->blob_top_label_->cpu_data()[0], 1);
  EXPECT_EQ(this->blob_top_label_->cpu_data()[1], 0);
  // cat gray.jpg
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 1);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  EXPECT_EQ(this->blob_top_label_->cpu_data()[0], 0);
  EXPECT_EQ(this->blob_top_label_->cpu_data()[1], 1);
}

}  // namespace caffe
#endif  // USE_OPENCV
