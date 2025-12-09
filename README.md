# autoencoder
Hiện tại là nhánh này đã có gpu-autoencoder. Tuy nhiên khi đem so với cpu-autoencoder thì output chưa giống. Setup so sánh như sau:  
+ 2 model load cùng một tập random weight 
+ setup lại batchSize =3.  
+ Chạy backprop 5 lần rồi dừng vòng lặp  
+ Xuất weight 2 model ra rồi so sánh từng cái.  

Mô tả các hàm:
+ `cpu-train.cpp`: train cpu-autoencoder
+ `gpu-train.cpp`: train gpu-autoencoder
+ `compare_bins.cpp`: so sánh 2 tập weight 