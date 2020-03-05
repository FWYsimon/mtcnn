
#include <io.h>
#include <fcntl.h>
#include <fstream>  // NOLINT(readability/streams)
#include "caffe.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/text_format.h>
#include <sys/stat.h>
#include <iosfwd>
#include <stdio.h>
#include <iostream>
#include <cblas.h>
#include <memory>
#include <stdarg.h>
#include <cv.h>
#include <highgui.h>

using namespace cv;

#ifdef _DEBUG
#pragma comment(lib, "libprotobufd.lib")
#else
#pragma comment(lib, "libprotobuf.lib")
#endif


#pragma comment(lib, "libopenblas.dll.a")

using namespace caffe;
using namespace std;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::ZeroCopyInputStream;

string sformat(const char* fmt, ...){
	va_list vl;
	va_start(vl, fmt);

	char buffer[10000];
	vsprintf(buffer, fmt, vl);
	return buffer;
}

void check(bool op, const char* file, int line, const char* code, const char* msgfmt, ...){

	if (op) return;
	
	va_list vl;
	va_start(vl, msgfmt);
	printf("CHECK fail( %s ): %s:%d: ", code, file, line);
	vprintf(msgfmt, vl);
	abort();
}

#ifdef _DEBUG
#define CHECK(op, msgfmt, ...)	check((op), __FILE__, __LINE__, #op, msgfmt, __VA_ARGS__)
#else
#define CHECK
#endif

template<typename _T>
void setzero(int count, _T* ptr){
	memset(ptr, 0, sizeof(_T) * count);
}

template<typename _T>
void copyto(_T* dst, _T* src, int count){
	memcpy(dst, src, sizeof(_T) * count);
}

struct Tensor{

	int n = 0;
	int c = 0;
	int h = 0;
	int w = 0;
	vector<float> d;

	Tensor(){}
	Tensor(int n, int c, int h, int w){
		this->n = n;
		this->c = c;
		this->h = h;
		this->w = w;
		resize(n, c, h, w);
	}

	void copyFrom(const shared_ptr<Tensor>& other){

		resizeLike(other);
		copyto(ptr(), other->ptr(), count());
	}

	void resizeByCount(int count){

		if (count < d.size())
			return;

		d.resize(count);
	}

	string shape(){ return sformat("%dx%dx%dx%d", n, c, h, w); }

	void resizeLike(const shared_ptr<Tensor>& other){

		resize(other->n, other->c, other->h, other->w);
	}

	void resize(int n = -1, int c = -1, int h = -1, int w = -1){

		n = n == -1 ? this->n : n;
		c = c == -1 ? this->c : c;
		h = h == -1 ? this->h : h;
		w = w == -1 ? this->w : w;

		CHECK(n * c * h * w > 0, "%d > 0", n * c * h * w);
		this->n = n;
		this->c = c;
		this->h = h;
		this->w = w;
		resizeByCount(count());
	}

	int count(int start_axis = 0){

		int dims[] = { n, c, h, w };
		int num_dims = sizeof(dims) / sizeof(dims[0]);
		CHECK(start_axis >= 0 && start_axis < num_dims, "start_axis(%d) >= 0 && start_axis(%d) < num_dims(%d)", start_axis, start_axis, num_dims);
		
		int _count = 1;
		for (int i = start_axis; i < num_dims; ++i)
			_count *= dims[i];
		return _count;
	}

	inline int offset(int n = 0, int c = 0, int h = 0, int w = 0){

		CHECK(n < this->n && n >= 0, "%d < %d", n, this->n);
		CHECK(c < this->c && c >= 0, "%d < %d", c, this->c);
		CHECK(h < this->h && h >= 0, "%d < %d", h, this->h);
		CHECK(w < this->w && w >= 0, "%d < %d", w, this->w);
		return (w + (h + (c + (n)* this->c) * this->h) * this->w);
	}

	void reshape(int n = -1, int c = -1, int h = -1, int w = -1){
		
		n = n == -1 ? this->n : n;
		c = c == -1 ? this->c : c;
		h = h == -1 ? this->h : h;
		w = w == -1 ? this->w : w;

		CHECK(n * c * h * w == count() && n * c * h * w > 0, "%d == %d && %d > 0", n * c * h * w, count(), n * c * h * w);
		this->n = n;
		this->c = c;
		this->h = h;
		this->w = w;
	}

	inline float& at(int n = 0, int c = 0, int h = 0, int w = 0){
		return *(d.data() + offset(n, c, h, w));
	}

	inline float* ptr(int n = 0, int c = 0, int h = 0, int w = 0){
		return d.data() + offset(n, c, h, w);
	}
};

struct Layer;
struct Blob{

	int id = 0;
	static int idd;
	string name;
	shared_ptr<Layer> owner;
	vector<shared_ptr<Layer>> outputto;
	shared_ptr<Tensor> data;

	string shape(){ return data->shape(); }
	void setid(){ id = idd++; }

	Blob(){ setid(); }
	Blob(const shared_ptr<Tensor>& data) :data(data){ setid(); }
};
int Blob::idd = 1;

struct Layer;
typedef Layer* (*createLayerInterface)();
#define RegisterLayer(name)			Layer::layerFactory[#name] = name::create
#define SetupLayerFunc(name)		static Layer* create(){return new name();}
#define CreateLayer(type)			Layer::newLayer(type)

struct Layer{

	string name;
	string type;
	vector<shared_ptr<Tensor>> params;
	vector<shared_ptr<Blob>> input, output;
	caffe::LayerParameter layer_param;
	
	static map<string, createLayerInterface> layerFactory;
	static Layer* newLayer(const string& type){

		string _type = type + "Layer";
		if (layerFactory.find(_type) == layerFactory.end())
			_type = "UnsupportLayer";
		return layerFactory[_type]();
	}

	virtual void reshape() = 0;
	virtual void forward() = 0;
};

map<string, createLayerInterface> Layer::layerFactory;

struct Im2colLayer : public Layer{

	SetupLayerFunc(Im2colLayer);

	virtual void reshape(){

		auto p = layer_param.convolution_param();
		auto in = input[0]->data;
		int kernel_size = p.kernel_size(0);
		int ksize = kernel_size * kernel_size;
		int rows = in->h;
		int cols = in->w;
		int num_output = p.num_output();
		int padding = p.pad_size() > 0 ? p.pad(0) : 0;
		int channels = in->c;
		int stride = p.stride_size() > 0 ? p.stride(0) : 0;
		int numCols = (cols - kernel_size + 1) * (rows - kernel_size + 1);
		output[0]->data->resize(in->n, 1, ksize * channels, numCols);
	}

	virtual void forward(){

		auto p = layer_param.convolution_param();
		auto in = input[0]->data;
		int kernel_size = p.kernel_size(0);
		int num_output = p.num_output();
		int colh = kernel_size * kernel_size;
		int ksize = kernel_size * kernel_size;
		int rows = in->h;
		int cols = in->w;
		int channels = in->c;
		auto& im2col_ = output[0]->data;

		//im2col
		for (int n = 0; n < in->n; ++n){
			for (int c = 0; c < in->c; ++c){

				int column_index = 0;
				for (int y = 0; y < rows - kernel_size + 1; ++y){
					for (int x = 0; x < cols - kernel_size + 1; ++x){

						for (int iy = 0; iy < kernel_size; ++iy){

							float* dataptr = in->ptr(n, c, y + iy, x);
							for (int ix = 0; ix < kernel_size; ++ix){

								float& colptr = im2col_->at(n, 0, c * ksize + iy * kernel_size + ix, column_index);
								colptr = *dataptr++;
							}
						}
						column_index++;
					}
				}
			}
		}
	}
};

struct ConvolutionLayer : public Layer{

	SetupLayerFunc(ConvolutionLayer);

	shared_ptr<Tensor> im2col_{ new Tensor() };
	shared_ptr<Tensor> kernel2col_{ new Tensor() };

	virtual void reshape(){

		auto p = layer_param.convolution_param();
		auto in = input[0]->data;
		int kernel_size = p.kernel_size(0);
		int ksize = kernel_size * kernel_size;
		int rows = in->h;
		int cols = in->w;
		int num_output = p.num_output();
		int padding = p.pad_size() > 0 ? p.pad(0) : 0;
		int channels = in->c;
		int stride = p.stride_size() > 0 ? p.stride(0) : 1;
		int numCols = (cols - kernel_size + 1) * (rows - kernel_size + 1);
		im2col_->resize(in->n, 1, ksize * channels, numCols);

		int out_width = (in->w + 2 * padding - kernel_size) / stride + 1;
		int out_height = (in->h + 2 * padding - kernel_size) / stride + 1;
		CHECK(padding == 0, "unsupport padding(%d) > 0", padding);

		output[0]->data->resize(in->n, num_output, out_height, out_width);

		//kernel2col
		kernel2col_->resize(1, 1, num_output, ksize * channels);

		auto& kernel = params[0];
		CHECK(kernel->count() == kernel2col_->count(), "count must same");
		copyto(kernel2col_->ptr(), kernel->ptr(), kernel2col_->count());
	}

	virtual void forward(){ 

		auto p = layer_param.convolution_param();
		auto in = input[0]->data;
		int kernel_size = p.kernel_size(0);
		int num_output = p.num_output();
		int colh = kernel_size * kernel_size;
		int ksize = kernel_size * kernel_size;
		int rows = in->h;
		int cols = in->w;
		int channels = in->c;

		if (kernel_size == 1){
			copyto(im2col_->ptr(), in->ptr(), in->count());
		}
		else{
			//im2col
			for (int n = 0; n < in->n; ++n){
				for (int c = 0; c < in->c; ++c){

					int column_index = 0;
					for (int y = 0; y < rows - kernel_size + 1; ++y){
						for (int x = 0; x < cols - kernel_size + 1; ++x){

							float* dataptr = in->ptr(n, c, y, x);
							float* colptr = im2col_->ptr(n, 0, c * ksize) + column_index;
							for (int iy = 0; iy < kernel_size; ++iy){

								for (int ix = 0; ix < kernel_size; ++ix){
									*colptr = *dataptr++;
									colptr += im2col_->w;
								}
								dataptr += in->w - kernel_size;
							}
							column_index++;
						}
					}
				}
			}
		}

		auto& bias = params[1];
		auto& outtensor = output[0]->data;
		for (int n = 0; n < im2col_->n; ++n){
			
			// kernel * im2col
			// output * (ksize * input) ---X--- (ksize * input) * [(in.w - kernel + 1) * (in.h - kernel + 1)]
			// result = output * [(in.w - kernel + 1) * (in.h - kernel + 1)]

			int a_rows = kernel2col_->h;
			int a_cols = kernel2col_->w;
			int b_rows = im2col_->h;
			int b_cols = im2col_->w;
			float alpha = 1;
			float beta = 0;
			int c_rows = a_rows;
			int c_cols = b_cols;
			float* aptr = kernel2col_->ptr();
			float* bptr = im2col_->ptr(n);
			float* cptr = outtensor->ptr(n);

			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
				a_rows,
				b_cols,
				a_cols,
				alpha,
				aptr,
				a_cols,
				bptr,
				b_cols,
				beta,
				cptr,
				c_cols);
		}

		for (int n = 0; n < outtensor->n; ++n){
			for (int c = 0; c < outtensor->c; ++c){
				float value = bias->d[c];
				float* ptr = outtensor->ptr(n, c);
				int count = outtensor->count(2);
				while (count--)
					*ptr++ += value;
			}
		}
	}
};

struct PReLULayer : public Layer{

	SetupLayerFunc(PReLULayer);

	virtual void reshape(){

		if (output[0] == input[0])
			return;

		output[0]->data->resizeLike(input[0]->data);
	}

	virtual void forward(){

		float* slope_data = params[0]->ptr();

		auto& in = input[0]->data;
		auto& out = output[0]->data;
		CHECK(in->count() == out->count(), "%d == %d", in->count(), out->count());

		int count = in->count();
		float* ptrin = in->ptr();
		float* ptrout = out->ptr();
		int dim = out->count(2);
		int channels = out->c;

		for (int i = 0; i < count; ++i){

			int c = (i / dim) % channels;
			*ptrout = std::max(*ptrin, 0.0f) + slope_data[c] * std::min(*ptrin, 0.0f);

			ptrin++;
			ptrout++;
		}
	}
};

struct PoolingLayer : public Layer{

	SetupLayerFunc(PoolingLayer);

	virtual void reshape(){

		//same卷积
		auto in = input[0]->data;
		auto p = layer_param.pooling_param();
		int stride = p.stride();
		int kernel_size = p.kernel_size();
		output[0]->data->resize(in->n, in->c, 
			ceil((in->h - kernel_size) / (float)stride) + 1, 
			ceil((in->w - kernel_size) / (float)stride) + 1);
	}

	virtual void forward(){

		auto& p = layer_param.pooling_param();
		const auto& type = p.pool();
		int kernel_size = p.kernel_size();
		int stride = p.stride();
		//

		auto& in = input[0]->data;
		auto& out = output[0]->data;
		int poolw = out->w;
		int poolh = out->h;

		setzero(out->count(), out->ptr());
		if (type == caffe::PoolingParameter_PoolMethod_AVE){
			for (int n = 0; n < in->n; ++n){
				for (int c = 0; c < in->c; ++c){
					for (int y = 0; y < poolh; ++y){
						for (int x = 0; x < poolw; ++x){

							int ys = y * stride;
							int xs = x * stride;
							int yend = min(ys + kernel_size, in->h);
							int xend = min(xs + kernel_size, in->w);
							int pool_size = (xend - xs) * (yend - ys);

							ys = max(ys, 0);
							xs = max(xs, 0);
							float value = 0;
							for (int iy = ys; iy < yend; ++iy){
								for (int ix = xs; ix < xend; ++ix)
									value += in->at(n, c, iy, ix);
							}
							out->at(n, c, y, x) = value /= pool_size;
						}
					}
				}
			}
		}
		else if (type == caffe::PoolingParameter_PoolMethod_MAX){

			float* dstval = out->ptr();
			for (int n = 0; n < in->n; ++n){
				for (int c = 0; c < in->c; ++c){
					for (int y = 0; y < poolh; ++y){
						for (int x = 0; x < poolw; ++x){
							int ys = y * stride;
							int xs = x * stride;
							int yend = min(ys + kernel_size, in->h);
							int xend = min(xs + kernel_size, in->w);
							int pool_size = (xend - xs) * (yend - ys);

							ys = max(ys, 0);
							xs = max(xs, 0);
							float value = -INFINITY;
							float* p = in->ptr(n, c, ys) + xs;
							for (int iy = ys; iy < yend; ++iy){
								for (int ix = xs; ix < xend; ++ix)
									value = std::max(value, *p++);
								p += in->w - (xend - xs);
							}
							*dstval++ = value;
						}
					}
				}
			}
		}
	}
};

struct InnerProductLayer : public Layer{

	SetupLayerFunc(InnerProductLayer);

	virtual void reshape(){

		auto in = input[0]->data;
		auto p = layer_param.inner_product_param();
		int num_output = p.num_output();
		output[0]->data->resize(in->n, num_output, 1, 1);
	}

	virtual void forward(){

		auto in = input[0]->data;
		auto p = layer_param.inner_product_param();
		int num_output = p.num_output();
		float* weights = params[0]->ptr();		//128 * 576
		int weights_rows = params[0]->n;		//128
		int weights_cols = params[0]->c;		//576
		float* bais = params[1]->ptr();			//127 * 1
		auto out = output[0]->data;

		int dims = in->count(1);

		//576 * 1
		float* data = in->ptr();
		int data_rows = in->n;
		int data_cols = dims;

		//data * weights.T
		int a_rows = data_rows;
		int a_cols = data_cols;
		int b_rows = weights_rows;
		int b_cols = weights_cols;
		int b_elastic_cols = b_rows;		//转置的
		float alpha = 1;
		float beta = 0;
		int c_rows = a_rows;
		int c_cols = b_rows;		//因为b转置了，所以是之前的行数
		float* aptr = data;
		float* bptr = weights;
		float* cptr = out->ptr();

		//n * 128
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
			a_rows,
			b_elastic_cols,
			a_cols,
			alpha,
			aptr,
			a_cols,
			bptr,
			b_cols,
			beta,
			cptr,
			c_cols);

		for (int i = 0; i < in->n; ++i){
			cblas_saxpy(num_output, 1.0f, bais, 1, out->ptr(i), 1);
		}
	}
};

struct SoftmaxLayer : public Layer{

	SetupLayerFunc(SoftmaxLayer);

	virtual void reshape(){

		auto in = input[0]->data;
		output[0]->data->resizeLike(in);
	}

	virtual void forward(){

		auto& in = input[0]->data;
		auto& out = output[0]->data;

		for (int n = 0; n < in->n; ++n){
			for (int y = 0; y < out->h; ++y){
				for (int x = 0; x < out->w; ++x){

					float sum = 0;
					for (int c = 0; c < in->c; ++c){
						float val = in->at(n, c, y, x);
						val = exp(val);

						out->at(n, c, y, x) = val;
						sum += val;
					}
					
					for (int c = 0; c < in->c; ++c)
						out->at(n, c, y, x) /= sum;
				}
			}
		}
	}
};

struct DropoutLayer : public Layer{

	SetupLayerFunc(DropoutLayer);

	virtual void reshape(){

		auto in = input[0]->data;
		output[0]->data->resizeLike(in);
	}

	virtual void forward(){

		if (input[0] != output[0])
			output[0]->data->copyFrom(input[0]->data);
	}
};

struct UnsupportLayer : public Layer{

	SetupLayerFunc(UnsupportLayer);

	virtual void reshape(){
		
		CHECK(false, "Unsupport layer, name = %s, type = %s\n", name.c_str(), type.c_str());
	}

	virtual void forward(){ 

		CHECK(false, "Unsupport layer, name = %s, type = %s\n", name.c_str(), type.c_str());
	}
};

struct CNN{

	vector<shared_ptr<Layer>> executionGraph;
	map<string, shared_ptr<Layer>> weights;
	map<string, shared_ptr<Blob>> blobs;

	bool loadModel(const string& deployFile, const string& caffemodelFile);

	void setup(){

		for (int i = 0; i < executionGraph.size(); ++i){
			for (int j = 0; j < executionGraph[i]->input.size(); ++j)
				blobs[executionGraph[i]->input[j]->name] = executionGraph[i]->input[j];

			for (int j = 0; j < executionGraph[i]->output.size(); ++j)
				blobs[executionGraph[i]->output[j]->name] = executionGraph[i]->output[j];
		}
	}

	shared_ptr<Tensor> input(){
		return executionGraph[0]->input[0]->data;
	}

	shared_ptr<Blob> blob(const string& name){
		if (blobs.find(name) == blobs.end())
			return shared_ptr<Blob>();

		return blobs[name];
	}

	void reshape(){
		for (int i = 0; i < executionGraph.size(); ++i)
			executionGraph[i]->reshape();
	}

	void forward(){

		if (executionGraph.empty())
			return;

		reshape();
		for (int i = 0; i < executionGraph.size(); ++i){

			auto& layer = executionGraph[i];
			//printf("%s[%s], input[%d]: %s, output[%d]: %s\n", layer->name.c_str(), layer->type.c_str(), layer->input[0]->id, layer->input[0]->shape().c_str(), layer->output[0]->id, layer->output[0]->shape().c_str());
			layer->forward();
		}
	}
};

int shapeDim(const BlobProto& blob, int axis){

	const auto& shape = blob.shape();
	if (axis < shape.dim_size())
		return shape.dim(axis);
	return 1;
}

bool loadMessageString(const string& file, NetParameter& net){

	int fd = _open(file.c_str(), O_RDONLY | O_BINARY);
	if (fd == -1){
		printf("open file[%s] fail.\n", file.c_str());
		return false;
	}

	std::shared_ptr<ZeroCopyInputStream> raw_input = std::make_shared<FileInputStream>(fd);
	bool success = google::protobuf::TextFormat::Parse(raw_input.get(), &net);
	raw_input.reset();
	_close(fd);
	if (!success){
		printf("Parse fail: %s\n", file.c_str());
		return false;
	}
	return true;
}

bool loadMessageBinary(const string& file, NetParameter& net){

	int fd = _open(file.c_str(), O_RDONLY | O_BINARY);
	if (fd == -1){
		printf("open file[%s] fail.\n", file.c_str());
		return false;
	}

	std::shared_ptr<ZeroCopyInputStream> raw_input = std::make_shared<FileInputStream>(fd);
	bool success = net.ParseFromZeroCopyStream(raw_input.get());
	raw_input.reset();
	_close(fd);
	if (!success){
		printf("Parse fail: %s\n", file.c_str());
		return false;
	}
	return true;
}

bool CNN::loadModel(const string& deployFile, const string& caffemodelFile){

	this->executionGraph.clear();
	this->weights.clear();

	NetParameter deploy, model;
	if (!loadMessageString(deployFile, deploy) || !loadMessageBinary(caffemodelFile, model))
		return false;

	map<string, shared_ptr<Blob>> ioblob;
	auto getblob = [&](const string& name){

		if (ioblob.find(name) == ioblob.end()){
			ioblob[name] = shared_ptr<Blob>(new Blob());
			ioblob[name]->data.reset(new Tensor());
		}

		return ioblob[name];
	};

	for (int i = 0; i < model.layer_size(); ++i){
		const LayerParameter& layer = model.layer(i);

		shared_ptr<Layer> l(CreateLayer(layer.type()));
		l->name = layer.name();
		l->type = layer.type();
		this->weights["%s", l->name.c_str()] = l;

		printf("layer: %s\n", layer.name().c_str());
		for (int j = 0; j < layer.bottom_size(); ++j){
			printf("bottom_%d: %s\n", j, layer.bottom(j).c_str());

			shared_ptr<Blob> blob(new Blob());
			blob->name = layer.bottom(j);
			blob->owner = l;
			l->input.push_back(blob);
		}
		
		for (int j = 0; j < layer.top_size(); ++j){
			printf("top_%d: %s\n", j, layer.top(j).c_str());

			shared_ptr<Blob> blob(new Blob());
			blob->name = layer.top(j);
			blob->owner = l;
			l->output.push_back(blob);
		}

		for (int j = 0; j < layer.blobs_size(); ++j){
			const BlobProto& params = layer.blobs(j);
			printf("param %d[%dx%dx%dx%d]: ", j, shapeDim(params, 0), shapeDim(params, 1), shapeDim(params, 2), shapeDim(params, 3));

			shared_ptr<Tensor> param(new Tensor(shapeDim(params, 0), shapeDim(params, 1), shapeDim(params, 2), shapeDim(params, 3)));
			l->params.push_back(param);

			int num = params.data_size();
			for (int k = 0; k < num; ++k)
				param->d[k] = params.data(k);

			num = min(10, num);
			for (int k = 0; k < num; ++k){
				if (k < num - 1)
					printf("%.2f,", params.data(k));
				else
					printf("%.2f", params.data(k));
			}

			if (num < params.data_size())
				printf("...");

			printf("\n");
		}
	}

	if (deploy.input_size() == 1){

		string name = deploy.input(0);
		shared_ptr<Blob> blob = getblob(name);

		int n = deploy.input_dim(0);
		int c = deploy.input_dim(1);
		int h = deploy.input_dim(2);
		int w = deploy.input_dim(3);

		blob->name = name;
		blob->data.reset(new Tensor(n, c, h, w));
	}

	Blob::idd = 1;
	for (int i = 0; i < deploy.layer_size(); ++i){
		const LayerParameter& layer = deploy.layer(i);

		shared_ptr<Layer> l(CreateLayer(layer.type()));
		if (this->weights.find(layer.name()) != this->weights.end())
			l->params = this->weights[layer.name()]->params;

		l->name = layer.name();
		l->type = layer.type();
		l->layer_param = layer;
		this->executionGraph.push_back(l);

		printf("layer: %s\n", layer.name().c_str());
		for (int j = 0; j < layer.bottom_size(); ++j){

			string name = layer.bottom(j);
			printf("bottom_%d: %s\n", j, name.c_str());
			
			shared_ptr<Blob> blob = getblob(name);
			blob->name = name;
			blob->outputto.push_back(l);
			l->input.push_back(blob);
		}

		for (int j = 0; j < layer.top_size(); ++j){

			string name = layer.top(j);
			printf("top_%d: %s\n", j, name.c_str());

			shared_ptr<Blob> blob = getblob(name);
			blob->name = name;
			blob->owner = l;
			l->output.push_back(blob);
		}
	}

	this->setup();
	return true;
}

struct FaceObject{

	float x, y, r, b;
	float score;
	vector<Point2f> keypoints;

	FaceObject(float x, float y, float r, float b, float score){
		this->x = x;
		this->y = y;
		this->r = r;
		this->b = b;
		this->score = score;
	}

	float area() const{
		return (r - x + 1) * (b - y + 1);
	}

	Rect box() const{
		return Rect(Point(x, y), Point((r), (b)));
	}

	float widthf() const{
		return r - x + 1;
		//return box().width;
	}

	float heightf() const{
		return b - y + 1;
		//return box().height;
	}

	Rect transbox() const{
		return Rect(Point(y, x), Point(b, r));
	}
};

float IoU(const FaceObject& a, const FaceObject& b){
	float xmax = max(a.x, b.x);
	float ymax = max(a.y, b.y);
	float xmin = min(a.r, b.r);
	float ymin = min(a.b, b.b);

	//Union
	float uw = (xmin - xmax + 1 > 0) ? (xmin - xmax + 1) : 0;
	float uh = (ymin - ymax + 1 > 0) ? (ymin - ymax + 1) : 0;
	float iou = uw * uh;

	//if (type == NMSType_IOUMin)
	//	return iou / min(a.area(), b.area());
	//else
	return iou / (a.area() + b.area() - iou);
}

void nms(vector<FaceObject>& objs, float threshold){

	std::sort(objs.begin(), objs.end(), [](FaceObject& a, FaceObject& b){
		return a.score > b.score;
	});

	for (int i = 0; i < objs.size(); ++i){

		for (int j = i + 1; j < objs.size(); ++j){

			float iou = IoU(objs[i], objs[j]);

			if (iou > threshold){
				objs.erase(objs.begin() + j);
				j--;
			}
		}
	}
}

class MTCNN{

public:
	MTCNN(){

		pnet.reset(new CNN());
		rnet.reset(new CNN());
		onet.reset(new CNN());

		pnet->loadModel("models/det1.prototxt", "models/det1.caffemodel");
		rnet->loadModel("models/det2.prototxt", "models/det2.caffemodel");
		onet->loadModel("models/det3.prototxt", "models/det3.caffemodel");
	}

	vector<FaceObject> detect(const Mat& frame, int min_face){

		forwardPNet(frame, min_face);
		forwardRNet();
		forwardONet();

		return onetoutput_;
	}

	vector<FaceObject> pnetResult(){
		return pnetoutput_;
	}

	vector<FaceObject> rnetResult(){
		return rnetoutput_;
	}

	vector<FaceObject> onetResult(){
		return onetoutput_;
	}

private:
	void forwardONet(){

		//"conv6-2", "prob1", "conv6-3"
		onetoutput_.clear();
		if (rnetoutput_.empty()) return;

		shared_ptr<Tensor> inputTensor = onet->input();
		Size inputSize(inputTensor->w, inputTensor->h);
		inputTensor->resize(rnetoutput_.size());

		for (int i = 0; i < rnetoutput_.size(); ++i){

			auto& obj = rnetoutput_[i];
			Rect roi = obj.box();
			roi = roi & Rect(0, 0, image_.cols, image_.rows);

			Mat inputImage = Mat::zeros(inputSize, CV_32FC3);
			if (roi.area() > 0){
				Mat imroi = image_(roi);
				resize(imroi, inputImage, inputSize);
			}

			Mat ms[3];
			for (int j = 0; j < 3; ++j)
				ms[j] = Mat(inputSize, CV_32F, inputTensor->ptr(i, j));
			split(inputImage, ms);
		}
		onet->forward();

		auto delta = onet->blob("conv6-2")->data;
		auto conf = onet->blob("prob1")->data;
		auto keypoints = onet->blob("conv6-3")->data;
		for (int i = 0; i < conf->n; ++i){

			float score = conf->at(i, 1);
			if (score > 0.7){

				float dy = delta->at(i, 0);
				float dx = delta->at(i, 1);
				float db = delta->at(i, 2);
				float dr = delta->at(i, 3);

				auto& target = rnetoutput_[i];
				FaceObject res = target;
				res.x = target.x + target.widthf() * dx;
				res.y = target.y + target.heightf() * dy;
				res.r = target.r + target.widthf() * dr;
				res.b = target.b + target.heightf() * db;

				float w = res.r - res.x + 1;
				float h = res.b - res.y + 1;
				float mxline = std::max(w, h);
				float cx = res.x + w * 0.5;
				float cy = res.y + h * 0.5;
				res.x = (cx - mxline * 0.5);
				res.y = (cy - mxline * 0.5);
				res.r = (res.x + mxline);
				res.b = (res.y + mxline);

				res.x = std::max(0.0f, std::min(res.x, image_.cols - 1.0f));
				res.y = std::max(0.0f, std::min(res.y, image_.rows - 1.0f));
				res.r = std::max(0.0f, std::min(res.r, image_.cols - 1.0f));
				res.b = std::max(0.0f, std::min(res.b, image_.rows - 1.0f));

				res.score = score;
				if ((res.r - res.x + 1) * (res.b - res.y + 1) > 0){

					for (int k = 0; k < keypoints->c / 2; ++k){
						float x = keypoints->at(i, k + 5);
						float y = keypoints->at(i, k);
						res.keypoints.push_back(Point2f(y * target.heightf() + target.y, x * target.widthf() + target.x));
					}
					onetoutput_.push_back(res);
				}
			}
		}
		nms(onetoutput_, 0.5);
	}

	void forwardRNet(){

		rnetoutput_.clear();
		if (pnetoutput_.empty()) return;

		//"conv5-2", "prob1"
		shared_ptr<Tensor> inputTensor = rnet->input();
		Size inputSize(inputTensor->w, inputTensor->h);
		inputTensor->resize(pnetoutput_.size());

		for (int i = 0; i < pnetoutput_.size(); ++i){

			auto& obj = pnetoutput_[i];
			Rect roi = obj.box();
			roi = roi & Rect(0, 0, image_.cols, image_.rows);

			Mat inputImage = Mat::zeros(inputSize, CV_32FC3);
			if (roi.area() > 0){
				Mat imroi = image_(roi);
				resize(imroi, inputImage, inputSize);
			}
			
			Mat ms[3];
			for (int j = 0; j < 3; ++j)
				ms[j] = Mat(inputSize, CV_32F, inputTensor->ptr(i, j));
			split(inputImage, ms);
		}
		rnet->forward();

		auto delta = rnet->blob("conv5-2")->data;
		auto conf = rnet->blob("prob1")->data;
		for (int i = 0; i < conf->n; ++i){

			float score = conf->at(i, 1);
			if (score > 0.7){

				float dy = delta->at(i, 0);
				float dx = delta->at(i, 1);
				float db = delta->at(i, 2);
				float dr = delta->at(i, 3);

				auto& target = pnetoutput_[i];
				FaceObject res = target;
				res.x = target.x + target.widthf() * dx;
				res.y = target.y + target.heightf() * dy;
				res.r = target.r + target.widthf() * dr;
				res.b = target.b + target.heightf() * db;

				float w = res.r - res.x + 1;
				float h = res.b - res.y + 1;
				float mxline = std::max(w, h);
				float cx = res.x + w * 0.5;
				float cy = res.y + h * 0.5;
				res.x = (cx - mxline * 0.5);
				res.y = (cy - mxline * 0.5);
				res.r = (res.x + mxline);
				res.b = (res.y + mxline);

				res.score = score;
				if ((res.r - res.x + 1) * (res.b - res.y + 1) > 0){
					rnetoutput_.push_back(res);
				}
			}
		}
		nms(rnetoutput_, 0.5);
	}

	void forwardPNet(const Mat& frame, int min_face){

		float m = 12.0 / min_face;
		float minl = min(frame.cols, frame.rows);
		float cur_scale = 1.0;
		float scale_factor = 0.709;

		all_scales_.clear();
		minl *= m;
		while (minl >= 12.0){
			all_scales_.push_back(m * cur_scale);
			cur_scale *= scale_factor;
			minl *= scale_factor;
		}

		Mat inputimage;
		cvtColor(frame, image_, CV_BGR2RGB);
		image_ = image_.t();
		image_.convertTo(image_, CV_32F, 1 / 127.5, -1.0);

		pnetoutput_.clear();
		for (int i = 0; i < all_scales_.size(); ++i){

			float scale = all_scales_[i];
			resize(image_, inputimage, Size(), scale, scale);

			shared_ptr<Tensor> inputTensor = pnet->input();
			inputTensor->resize(1, 3, inputimage.rows, inputimage.cols);

			Mat ms[3];
			for (int i = 0; i < 3; ++i)
				ms[i] = Mat(inputimage.rows, inputimage.cols, CV_32F, inputTensor->ptr(0, i));

			split(inputimage, ms);
			pnet->forward();
			decodePNet(scale);
		}
		nms(pnetoutput_, 0.5);
	}

	void decodePNet(float scale){

		auto delta = pnet->blob("conv4-2")->data;
		auto output = pnet->blob("prob1");
		auto& outdata = output->data;
		Mat omatrix(outdata->h, outdata->w, CV_32F, outdata->ptr(0, 1));

		int stride = 2;
		int cellsize = 12;
		vector<FaceObject> objs;
		for (int j = 0; j < omatrix.rows; ++j){
			for (int i = 0; i < omatrix.cols; ++i){

				float score = omatrix.at<float>(j, i);
				if (score > 0.8){

					float dy = delta->at(0, 0, j, i);
					float dx = delta->at(0, 1, j, i);
					float db = delta->at(0, 2, j, i);
					float dr = delta->at(0, 3, j, i);

					float x = (i * stride + dx * cellsize) / scale;
					float y = (j * stride + dy * cellsize) / scale;
					float r = (i * stride + dr * cellsize + cellsize - 1) / scale;
					float b = (j * stride + db * cellsize + cellsize - 1) / scale;

					//x = std::max(0.0f, std::min(x, image_.cols - 1.0f));
					//y = std::max(0.0f, std::min(y, image_.rows - 1.0f));
					//r = std::max(0.0f, std::min(r, image_.cols - 1.0f));
					//b = std::max(0.0f, std::min(b, image_.rows - 1.0f));

					float w = r - x + 1;
					float h = b - y + 1;
					float mxline = std::max(w, h);
					float cx = x + w * 0.5;
					float cy = y + h * 0.5;
					x = (cx - mxline * 0.5);
					y = (cy - mxline * 0.5);
					r = (x + mxline);
					b = (y + mxline);

					if (w * h > 0){
						objs.push_back(FaceObject(x, y, r, b, score));
					}
				}
			}
		}
		nms(objs, 0.5);
		pnetoutput_.insert(pnetoutput_.end(), objs.begin(), objs.end());
	}

private:
	Mat image_;
	vector<FaceObject> pnetoutput_;
	vector<FaceObject> rnetoutput_;
	vector<FaceObject> onetoutput_;
	vector<float> all_scales_;
	shared_ptr<CNN> pnet, rnet, onet;
};

int main(int argc, char** argv){

	RegisterLayer(UnsupportLayer);
	RegisterLayer(ConvolutionLayer);
	RegisterLayer(PReLULayer);
	RegisterLayer(PoolingLayer);
	RegisterLayer(SoftmaxLayer);
	RegisterLayer(InnerProductLayer);
	RegisterLayer(DropoutLayer);
	RegisterLayer(Im2colLayer);

	if (1)
	{
		CNN pnet;
		pnet.loadModel("models/det1.prototxt", "models/det1.caffemodel");

		Mat im = imread("imgs/t.jpg");
		Mat inputimage;
		cvtColor(im, inputimage, CV_BGR2RGB);
		inputimage = inputimage.t();
		inputimage.convertTo(inputimage, CV_32F, 1 / 127.5, -1.0);

		shared_ptr<Tensor> inputTensor = pnet.input();
		inputTensor->resize(1, 3, inputimage.rows, inputimage.cols);

		Mat ms[3];
		for (int i = 0; i < 3; ++i)
			ms[i] = Mat(inputimage.rows, inputimage.cols, CV_32F, inputTensor->ptr(0, i));

		split(inputimage, ms);
		pnet.forward();

		auto conv1 = pnet.blob("conv1")->data;
		Mat matrix(conv1->h * conv1->c, conv1->w, CV_32F, conv1->ptr());

		exit(0);
	};

	MTCNN mtcnn;

#if 0
	VideoCapture cap(0);
	Mat im;

	if (!cap.isOpened()){
		printf("can not open.\n");
		return 0;
	}

	while (!cap.read(im));
	while (cap.read(im)){

		double tick = getTickCount();
		mtcnn.detect(im, 80);
		tick = (getTickCount() - tick) / getTickFrequency() * 1000;
		printf("耗时：%.2f ms\n", tick);

		string names[] = { "PNet", "RNet", "ONet" };
		for (int k = 0; k < 3; ++k){

			Mat show = im.clone();
			vector<FaceObject> objs;

			if (k == 0){
				objs = mtcnn.pnetResult();
			}
			else if (k == 1){
				objs = mtcnn.rnetResult();
			}
			else if (k == 2){
				objs = mtcnn.onetResult();
			}

			for (int i = 0; i < objs.size(); ++i){
				rectangle(show, objs[i].transbox(), Scalar(0, 255));

				if (k == 2){
					for (int m = 0; m < objs[i].keypoints.size(); ++m){
						circle(show, objs[i].keypoints[m], 3, Scalar(0, 255), -1);
					}
				}
			}
			imshow(names[k], show);

		}
		waitKey(1);
	}
#else
	
	Mat im = imread("imgs/cmj2.jpg");

	double tick = getTickCount();
	auto objs = mtcnn.detect(im, 12);

	tick = (getTickCount() - tick) / getTickFrequency() * 1000;
	printf("耗时：%.2f ms\n", tick);

	for (auto& o : objs){
		rectangle(im, o.transbox(), Scalar(0, 255), 2);
	
		for (int i = 0; i < o.keypoints.size(); ++i)
			circle(im, o.keypoints[i], 3, Scalar(0, 0, 255), -1);
	}

	//string names[] = { "PNet", "RNet", "ONet" };
	//for (int k = 0; k < 3; ++k){
	//
	//	Mat show = im.clone();
	//	vector<FaceObject> objs;
	//
	//	if (k == 0){
	//		objs = mtcnn.pnetResult();
	//	}
	//	else if (k == 1){
	//		objs = mtcnn.rnetResult();
	//	}
	//	else if (k == 2){
	//		objs = mtcnn.onetResult();
	//	}
	//
	//	for (int i = 0; i < objs.size(); ++i){
	//		rectangle(show, objs[i].transbox(), Scalar(0, 255));
	//
	//		if (k == 2){
	//			for (int m = 0; m < objs[i].keypoints.size(); ++m){
	//				circle(show, objs[i].keypoints[m], 3, Scalar(0, 255), -1);
	//			}
	//		}
	//	}
	//	imshow(names[k], show);
	//
	//}

	imshow("demo", im);
	waitKey();

#endif
	return 0;
}