# Đặt phiên bản CMake tối thiểu
cmake_minimum_required(VERSION 3.10)

# Đặt tên dự án và chỉ định ngôn ngữ là C++
project(TensorProject CXX)

# Đặt tiêu chuẩn C++ (ví dụ: C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Thêm thư mục 'src' vào đường dẫn tìm kiếm header
include_directories(src)

# Tự động tìm tất cả các tệp .cpp trong thư mục 'src' EXCEPT compare_latent.cpp (it has its own main)
file(GLOB SRC_FILES "src/nn/autoencoder.cpp" "src/nn/utils.cpp")

# Tạo tệp thực thi 'main' từ main.cpp và tất cả các tệp nguồn đã tìm thấy.
add_executable(main main.cpp ${SRC_FILES} "src/features_extractor.cpp" "src/dataset.cpp")
add_executable(cpu_train cpu_train.cpp ${SRC_FILES} "src/features_extractor.cpp" "src/dataset.cpp")

# Create separate executable for compare_latent (it has its own main)
add_executable(compare_latent "src/nn/compare_latent.cpp")

message("Project root: ${CMAKE_SOURCE_DIR}")
message("Current dir: ${CMAKE_CURRENT_SOURCE_DIR}")
# Tạo tệp thực thi cho test gradient
# add_executable(test_gradient test_gradient.cpp ${SRC_FILES})