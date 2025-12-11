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

#### Cập nhật 11/12:
Optimize hàm backward cho weight và input theo hướng sử dụng shared memory.
Thời gian không cải thiện. Kết quả ra vẫn đúng. Khả năng là do sai:
```
=== Time summary ===
Total Kernel Time: 443458.92 ms
Convolution Forward Time: 39600.96 ms||| Ratio: 8.93%
Convolution Backward Time: 401333.87 ms||| Ratio: 90.50%
Convolution Time: 440934.83 ms||| Ratio: 99.43%
ReLU Time: 917.14 ms||| Ratio: 0.21%
Pooling Time: 1606.95 ms||| Ratio: 0.36%

=== GPU Training Completed ===
```