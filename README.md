# autoencoder
Hiện tại là nhánh này đã có gpu-autoencoder. Tuy nhiên khi đem so với cpu-autoencoder thì output chưa giống. Setup so sánh như sau:  
+ 2 model load cùng một tập random weight 
+ setup lại batchSize =3.  
+ Chạy backprop 5 lần rồi dừng vòng lặp  
+ Xuất weight 2 model ra rồi so sánh từng cái.  

Mô tả các hàm:
+ `cpu-train.cpp`: train cpu-autoencoder, xuất ra file weights/cpu_trained_weights.bin
+ `gpu-train.cpp`: train gpu-autoencoder, xuất ra file weights/gpu_trained_weights.bin
+ `compare_bins.cpp`: so sánh 2 tập weight 

#### Cập nhật 10/12:
Đã fix bug. 2 bộ weights của cpu và gpu đã giống nhau sau vài lần backprop

Thời gian đo hiện tại:
```
=== Time summary ===
Total Kernel Time: 403288.14 ms
Convolution Forward Time: 39523.29 ms||| Ratio: 9.80%
Convolution Backward Time: 361250.60 ms||| Ratio: 89.58%
Convolution Time: 400773.89 ms||| Ratio: 99.38%
ReLU Time: 898.60 ms||| Ratio: 0.22%
Pooling Time: 1615.64 ms||| Ratio: 0.40%
```