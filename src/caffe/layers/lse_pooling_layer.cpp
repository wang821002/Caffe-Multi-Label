#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/lse_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void LsePoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LsePoolingParameter lse_pool_param = this->layer_param_.lse_pooling_param();
  if (lse_pool_param.global_pooling()) {
    CHECK(!(lse_pool_param.has_kernel_size() ||
      lse_pool_param.has_kernel_h() || lse_pool_param.has_kernel_w()))
      << "With Global_pooling: true Filter size cannot specified";
  } else {
    CHECK(!lse_pool_param.has_kernel_size() !=
      !(lse_pool_param.has_kernel_h() && lse_pool_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
    CHECK(lse_pool_param.has_kernel_size() ||
      (lse_pool_param.has_kernel_h() && lse_pool_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  }
  CHECK((!lse_pool_param.has_pad() && lse_pool_param.has_pad_h()
      && lse_pool_param.has_pad_w())
      || (!lse_pool_param.has_pad_h() && !lse_pool_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!lse_pool_param.has_stride() && lse_pool_param.has_stride_h()
      && lse_pool_param.has_stride_w())
      || (!lse_pool_param.has_stride_h() && !lse_pool_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  global_pooling_ = lse_pool_param.global_pooling();
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  } else {
    if (lse_pool_param.has_kernel_size()) {
      kernel_h_ = kernel_w_ = lse_pool_param.kernel_size();
    } else {
      kernel_h_ = lse_pool_param.kernel_h();
      kernel_w_ = lse_pool_param.kernel_w();
    }
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!lse_pool_param.has_pad_h()) {
    pad_h_ = pad_w_ = lse_pool_param.pad();
  } else {
    pad_h_ = lse_pool_param.pad_h();
    pad_w_ = lse_pool_param.pad_w();
  }
  if (!lse_pool_param.has_stride_h()) {
    stride_h_ = stride_w_ = lse_pool_param.stride();
  } else {
    stride_h_ = lse_pool_param.stride_h();
    stride_w_ = lse_pool_param.stride_w();
  }
  if (global_pooling_) {
    CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
      << "With Global_pooling: true; only pad = 0 and stride = 1";
  }
  if (pad_h_ != 0 || pad_w_ != 0) {
    CHECK(this->layer_param_.lse_pooling_param().pool()
        == LsePoolingParameter_PoolMethod_AVE
        || this->layer_param_.lse_pooling_param().pool()
        == LsePoolingParameter_PoolMethod_MAX
        || this->layer_param_.lse_pooling_param().pool()
        == LsePoolingParameter_PoolMethod_LSE)
        << "Padding implemented only for LSE, average and max pooling.";
    CHECK_LT(pad_h_, kernel_h_);
    CHECK_LT(pad_w_, kernel_w_);
  }
  r_T_ = lse_pool_param.r_t();
  std::cout << r_T_ << ' ' << lse_pool_param.r_t() << std::endl;
}

template <typename Dtype>
void LsePoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  }
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
  if (pad_h_ || pad_w_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
      --pooled_height_;
    }
    if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
      --pooled_width_;
    }
    CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
    CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
  }
  top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }

  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.lse_pooling_param().pool() ==
      LsePoolingParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
        pooled_width_);
  }
/*  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.lse_pooling_param().pool() ==
      LsePoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  }*/

}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void LsePoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.lse_pooling_param().pool()) {
  case LsePoolingParameter_PoolMethod_MAX:
    // Initialize
    if (use_top_mask) {
      top_mask = top[1]->mutable_cpu_data();
      caffe_set(top_count, Dtype(-1), top_mask);
    } else {
      mask = max_idx_.mutable_cpu_data();
      caffe_set(top_count, -1, mask);
    }
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_);
            int wend = min(wstart + kernel_w_, width_);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            const int pool_index = ph * pooled_width_ + pw;
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int index = h * width_ + w;
                if (bottom_data[index] > top_data[pool_index]) {
                  top_data[pool_index] = bottom_data[index];
                  if (use_top_mask) {
                    top_mask[pool_index] = static_cast<Dtype>(index);
                  } else {
                    mask[pool_index] = index;
                  }
                }
              }
            }
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  case LsePoolingParameter_PoolMethod_AVE:
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                top_data[ph * pooled_width_ + pw] +=
                    bottom_data[h * width_ + w];
              }
            }
            top_data[ph * pooled_width_ + pw] /= pool_size;
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
      }
    }
    break;
  case LsePoolingParameter_PoolMethod_LSE:
	for (int i = 0; i < top_count; ++i) {
	  top_data[i] = 0;
	}
	// The main loop
	for (int n = 0; n < bottom[0]->num(); ++n) {
	  for (int c = 0; c < channels_; ++c) {
		for (int ph = 0; ph < pooled_height_; ++ph) {
		  for (int pw = 0; pw < pooled_width_; ++pw) {
			int hstart = ph * stride_h_ - pad_h_;
			int wstart = pw * stride_w_ - pad_w_;
			int hend = min(hstart + kernel_h_, height_ + pad_h_);
			int wend = min(wstart + kernel_w_, width_ + pad_w_);
			int pool_size = (hend - hstart) * (wend - wstart);
			hstart = max(hstart, 0);
			wstart = max(wstart, 0);
			hend = min(hend, height_);
			wend = min(wend, width_);
			float maxTemp = 0;
			for (int h = hstart; h < hend; ++h) {
				for (int w = wstart; w < wend; ++w) {
					const int index = h * width_ + w;
					if (abs(bottom_data[index]) > maxTemp) {
						maxTemp = bottom_data[index];
					}
				}
			}
			for (int h = hstart; h < hend; ++h) {
			  for (int w = wstart; w < wend; ++w) {
				top_data[ph * pooled_width_ + pw] += 
						exp(r_T_ * (bottom_data[h * width_ + w] - maxTemp));
			  }
			}
			top_data[ph * pooled_width_ + pw] /= pool_size;
			top_data[ph * pooled_width_ + pw] = maxTemp + (1.0 / r_T_) *
					log(top_data[ph * pooled_width_ + pw]);
		  }
		}
		// compute offset
		bottom_data += bottom[0]->offset(0, 1);
		top_data += top[0]->offset(0, 1);
	  }
	}
	break;
  case LsePoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void LsePoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.lse_pooling_param().pool()) {
  case LsePoolingParameter_PoolMethod_MAX:
    // The main loop
    if (use_top_mask) {
      top_mask = top[1]->cpu_data();
    } else {
      mask = max_idx_.cpu_data();
    }
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            const int index = ph * pooled_width_ + pw;
            const int bottom_index =
                use_top_mask ? top_mask[index] : mask[index];
            bottom_diff[bottom_index] += top_diff[index];
          }
        }
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  case LsePoolingParameter_PoolMethod_AVE:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                bottom_diff[h * width_ + w] +=
                  top_diff[ph * pooled_width_ + pw] / pool_size;
              }
            }
          }
        }
        // offset
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case LsePoolingParameter_PoolMethod_LSE:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
//            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            float maxTemp = 0;
            float expSum = 0;
            for (int h = hstart; h < hend; ++h) {
            	for (int w = wstart; w < wend; ++w) {
            		const int index = h * width_ + w;
            		if (abs(bottom_data[index]) > maxTemp) {
            			maxTemp = bottom_data[index];
            		}
            	}
            }
            for (int h = hstart; h < hend; ++h) {
            	for (int w = wstart; w < wend; ++w) {
            		const int index = h * width_ + w;
            			expSum = expSum + exp(r_T_ * (bottom_data[index] - maxTemp));

            	}
            }
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
            	  const int index = h * width_ + w;
            	  bottom_diff[index] +=
            			  top_diff[ph * pooled_width_ + pw] *
            			  exp(r_T_ * (bottom_data[index] - maxTemp)) / expSum;
              }
            }
          }
        }
        // offset
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case LsePoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}


#ifdef CPU_ONLY
STUB_GPU(LsePoolingLayer);
#endif

INSTANTIATE_CLASS(LsePoolingLayer);
REGISTER_LAYER_CLASS(LsePooling);

}  // namespace caffe
