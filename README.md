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

#### Evaluation:

Setup: 
+ Batchsize = 32
+ epoch = 1
+ blocksize = 32

Ground truth loss (dùng để so sánh để xem kết quả có bị thay đổi không):

Processing batch 10/313 (samples 288 to 319)|   Single batch Loss: 0.297234
Processing batch 20/313 (samples 608 to 639)|   Single batch Loss: 0.229707
Processing batch 30/313 (samples 928 to 959)|   Single batch Loss: 0.229481
Processing batch 40/313 (samples 1248 to 1279)|   Single batch Loss: 0.215662
Processing batch 50/313 (samples 1568 to 1599)|   Single batch Loss: 0.202848

#### Cập nhật 10/12:
Đã fix bug. 2 bộ weights của cpu và gpu đã giống nhau sau vài lần backprop

Thời gian đo hiện tại:
```
=== Time summary ===
Total Kernel Time: 198470.66 ms
Convolution Forward Time: 19456.93 ms||| Ratio: 9.80%
Convolution Backward Time: 177719.29 ms||| Ratio: 89.54%
Convolution Time: 197176.22 ms||| Ratio: 99.35%
ReLU Time: 468.41 ms||| Ratio: 0.24%
Pooling Time: 826.04 ms||| Ratio: 0.42%
```

#### Cập nhật 11/12:
Optimize hàm backward cho weight và input theo hướng sử dụng shared memory.
Thời gian không cải thiện. Kết quả ra vẫn đúng. Khả năng là do sai:
```
=== Time summary ===
Total Kernel Time: 216439.65 ms
Convolution Forward Time: 18708.59 ms||| Ratio: 8.64%
Convolution Backward Time: 196452.94 ms||| Ratio: 90.77%
Convolution Time: 215161.53 ms||| Ratio: 99.41%
ReLU Time: 471.13 ms||| Ratio: 0.22%
Pooling Time: 806.99 ms||| Ratio: 0.37%
```

#### Cập nhật 12/12:
##### Refactor:
- Giữ lại kernel không tối ưu: `k_conv2d_backward_input`, `k_conv2d_backward_weights`
- Rename các kernel smem: `k_conv2d_backward_input_smem`, `k_conv2d_backward_weights_smem`



