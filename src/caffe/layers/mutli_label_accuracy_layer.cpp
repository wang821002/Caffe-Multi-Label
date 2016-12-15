#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/multi_label_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
//   top_k_ = this->layer_param_.accuracy_param().top_k();

//  has_ignore_label_ =
//    this->layer_param_.accuracy_param().has_ignore_label();
//  if (has_ignore_label_) {
//    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
//  }
//	CHECK_EQ(bottom[0]->num(), bottom[1]->num())
//	<< "The data and label should have the same number of instances";
//	CHECK_EQ(bottom[0]->channels(), bottom[1]->channels())
//	<< "The data and label should have the same number of channels";
//	CHECK_EQ(bottom[0]->height(), bottom[1]->height())
//	<< "The data and label should have the same height";
//	CHECK_EQ(bottom[0]->width(), bottom[1]->width())
//	<< "The data and label should have the same width";
	// Top will contain:
	// top[0] = Sensitivity or Recall (TP/P),
	// top[1] = Specificity (TN/N),
	// top[2] = Harmonic Mean of Sens and Spec, (2/(P/TP+N/TN))
	// top[3] = Precision (TP / (TP + FP))
	// top[4] = F1 Score (2 TP / (2 TP + FP + FN))
	//top[0]->Reshape(1, 5, 1, 1);
}

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	//LossLayer<Dtype>::Reshape(bottom, top);
	CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
			"MUTLI_LABEL_ACCURACY layer inputs must have the same count.";
	vector<int> top_shape(1);  // Accuracy is a scalar; 0 axes.
	top_shape[0] = 9;	
	top[0]->Reshape(top_shape);
//	if (top.size() > 1) {
//		// Per-class accuracy is a vector; 1 axes.
//		vector<int> top_shape_per_class(1);
//		top_shape_per_class[0] = bottom[0]->shape(label_axis_);
//		top[1]->Reshape(top_shape_per_class);
//		nums_buffer_.Reshape(top_shape_per_class);
//	}
}

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //Dtype accuracy = 0;
  Dtype true_positive = 0;
  Dtype false_positive = 0;
  Dtype true_negative = 0;
  Dtype false_negative = 0;
  int count_pos = 0;
  int count_neg = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
//  const int dim = bottom[0]->count() / outer_num_;
//  const int num_labels = bottom[0]->shape(label_axis_);
//  vector<Dtype> maxval(top_k_+1);
//  vector<int> max_id(top_k_+1);
//  if (top.size() > 1) {
//    caffe_set(nums_buffer_.count(), Dtype(0), nums_buffer_.mutable_cpu_data());
//    caffe_set(top[1]->count(), Dtype(0), top[1]->mutable_cpu_data());
//  }
//  int count = 0;
//  for (int i = 0; i < outer_num_; ++i) {
//    for (int j = 0; j < inner_num_; ++j) {
//      const int label_value =
//          static_cast<int>(bottom_label[i * inner_num_ + j]);
//      if (has_ignore_label_ && label_value == ignore_label_) {
//        continue;
//      }
//      if (top.size() > 1) ++nums_buffer_.mutable_cpu_data()[label_value];
//      DCHECK_GE(label_value, 0);
//      DCHECK_LT(label_value, num_labels);
//      // Top-k accuracy
//      std::vector<std::pair<Dtype, int> > bottom_data_vector;
//      for (int k = 0; k < num_labels; ++k) {
//        bottom_data_vector.push_back(std::make_pair(
//            bottom_data[i * dim + k * inner_num_ + j], k));
//      }
//      std::partial_sort(
//          bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
//          bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
//      // check if true label is in top k predictions
//      for (int k = 0; k < top_k_; k++) {
//        if (bottom_data_vector[k].second == label_value) {
//          ++accuracy;
//          if (top.size() > 1) ++top[1]->mutable_cpu_data()[label_value];
//          break;
//        }
//      }
//      ++count;
//    }
//  }
  
  int count = bottom[0]->count();

  for (int ind = 0; ind < count; ++ind) {
	  // Accuracy
	  int label = static_cast<int>(bottom_label[ind]);
	  if (label > 0.0001) {
		  // Update Positive accuracy and count
		  true_positive += (bottom_data[ind] >= 0.5);
		  false_negative += (bottom_data[ind] < 0.5);
		  count_pos++;
	  }
	  if (label < 0.0001) {
		  // Update Negative accuracy and count
		  true_negative += (bottom_data[ind] <0.5);
		  false_positive += (bottom_data[ind] >= 0.5);
		  count_neg++;
	  }
  }
  Dtype sensitivity = (count_pos > 0)? (true_positive / count_pos) : 0;
  Dtype specificity = (count_neg > 0)? (true_negative / count_neg) : 0;
  Dtype harmmean = ((count_pos + count_neg) > 0)?
		  2 / (count_pos / true_positive + count_neg / true_negative) : 0;
  Dtype precission = (true_positive > 0)?
		  (true_positive / (true_positive + false_positive)) : 0;
  Dtype f1_score = (true_positive > 0)?
		  2 * true_positive /
		  (2 * true_positive + false_positive + false_negative) : 0;

  DLOG(INFO) << "Sensitivity: " << sensitivity;
  DLOG(INFO) << "Specificity: " << specificity;
  DLOG(INFO) << "Harmonic Mean of Sens and Spec: " << harmmean;
  DLOG(INFO) << "Precission: " << precission;
  DLOG(INFO) << "F1 Score: " << f1_score;
  top[0]->mutable_cpu_data()[0] = sensitivity;
  top[0]->mutable_cpu_data()[1] = specificity;
  top[0]->mutable_cpu_data()[2] = harmmean;
  top[0]->mutable_cpu_data()[3] = precission;
  top[0]->mutable_cpu_data()[4] = f1_score;
  top[0]->mutable_cpu_data()[5] = true_positive;
  top[0]->mutable_cpu_data()[6] = false_negative;
  top[0]->mutable_cpu_data()[7] = true_negative;
  top[0]->mutable_cpu_data()[8] = false_positive;
  
//  // LOG(INFO) << "Accuracy: " << accuracy;
//  top[0]->mutable_cpu_data()[0] = accuracy / count;
//  if (top.size() > 1) {
//    for (int i = 0; i < top[1]->count(); ++i) {
//      top[1]->mutable_cpu_data()[i] =
//          nums_buffer_.cpu_data()[i] == 0 ? 0
//          : top[1]->cpu_data()[i] / nums_buffer_.cpu_data()[i];
//    }
//  }
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(MultiLabelAccuracyLayer);
REGISTER_LAYER_CLASS(MultiLabelAccuracy);

}  // namespace caffe
