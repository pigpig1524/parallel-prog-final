#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <iomanip>

class BinaryFileComparator {
public:
    static bool compareFiles(const std::string& file1, const std::string& file2, 
                           double tolerance = 1e-6f, bool verbose = true) {
        // Open files
        std::ifstream f1(file1, std::ios::binary);
        std::ifstream f2(file2, std::ios::binary);
        
        if (!f1.is_open()) {
            std::cerr << "Error: Cannot open file " << file1 << std::endl;
            return false;
        }
        
        if (!f2.is_open()) {
            std::cerr << "Error: Cannot open file " << file2 << std::endl;
            return false;
        }
        
        // Get file sizes
        f1.seekg(0, std::ios::end);
        f2.seekg(0, std::ios::end);
        
        std::size_t size1 = f1.tellg();
        std::size_t size2 = f2.tellg();
        
        if (size1 != size2) {
            std::cout << "Files have different sizes:" << std::endl;
            std::cout << "  " << file1 << ": " << size1 << " bytes" << std::endl;
            std::cout << "  " << file2 << ": " << size2 << " bytes" << std::endl;
            return false;
        }
        
        // Reset to beginning
        f1.seekg(0, std::ios::beg);
        f2.seekg(0, std::ios::beg);
        
        // Calculate number of double elements
        std::size_t numElements = size1 / sizeof(double);
        
        if (verbose) {
            std::cout << "Comparing " << numElements << " double elements..." << std::endl;
            std::cout << "Tolerance: " << tolerance << std::endl;
        }
        
        // Read and compare element by element
        std::vector<double> buffer1(1024), buffer2(1024);
        std::size_t elementsRead = 0;
        std::size_t differences = 0;
        double maxDiff = 0.0f;
        std::size_t maxDiffIndex = 0;
        
        while (elementsRead < numElements) {
            std::size_t elementsToRead = std::min(static_cast<std::size_t>(1024), 
                                                 numElements - elementsRead);
            
            f1.read(reinterpret_cast<char*>(buffer1.data()), 
                   elementsToRead * sizeof(double));
            f2.read(reinterpret_cast<char*>(buffer2.data()), 
                   elementsToRead * sizeof(double));
            
            if (f1.gcount() != f2.gcount()) {
                std::cerr << "Error reading files at element " << elementsRead << std::endl;
                return false;
            }
            
            // Compare elements in this chunk
            for (std::size_t i = 0; i < elementsToRead; ++i) {
                double diff = std::abs(buffer1[i] - buffer2[i]);
                
                if (diff > tolerance) {
                    differences++;
                    
                    // Print first few differences if verbose
                    if (verbose && differences <= 10) {
                        std::cout << "Difference at element " << (elementsRead + i) 
                                 << ": " << std::fixed << std::setprecision(8)
                                 << buffer1[i] << " vs " << buffer2[i] 
                                 << " (diff: " << diff << ")" << std::endl;
                    }
                }
                
                // Track maximum difference
                if (diff > maxDiff) {
                    maxDiff = diff;
                    maxDiffIndex = elementsRead + i;
                }
            }
            
            elementsRead += elementsToRead;
        }
        
        // Print summary
        if (verbose) {
            std::cout << "\n=== COMPARISON SUMMARY ===" << std::endl;
            std::cout << "Total elements: " << numElements << std::endl;
            std::cout << "Different elements: " << differences << std::endl;
            std::cout << "Accuracy: " << std::fixed << std::setprecision(4) 
                     << (100.0 * (numElements - differences) / numElements) << "%" << std::endl;
            std::cout << "Maximum difference: " << std::scientific << maxDiff 
                     << " at element " << maxDiffIndex << std::endl;
        }
        
        return differences == 0;
    }
    
    static void printFileInfo(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Cannot open " << filename << std::endl;
            return;
        }
        
        file.seekg(0, std::ios::end);
        std::size_t size = file.tellg();
        std::size_t numFloats = size / sizeof(double);
        
        std::cout << "File: " << filename << std::endl;
        std::cout << "  Size: " << size << " bytes" << std::endl;
        std::cout << "  Elements: " << numFloats << " doubles" << std::endl;
        
        // Read first few elements for preview
        file.seekg(0, std::ios::beg);
        std::vector<double> preview(std::min(numFloats, static_cast<std::size_t>(5)));
        file.read(reinterpret_cast<char*>(preview.data()), 
                 preview.size() * sizeof(double));
        
        std::cout << "  First elements: ";
        for (std::size_t i = 0; i < preview.size(); ++i) {
            std::cout << std::fixed << std::setprecision(6) << preview[i];
            if (i < preview.size() - 1) std::cout << ", ";
        }
        if (numFloats > 5) std::cout << " ...";
        std::cout << std::endl << std::endl;
    }
};

int main(int argc, char* argv[]) {
    
    std::string file1 = "weights/cpu_trained_weights.bin";
    std::string file2 = "weights/test_weights.bin";
    double tolerance = 1e-4f;

    
    std::cout << "=== BINARY FILE COMPARISON ===" << std::endl << std::endl;
    
    // Print file information
    BinaryFileComparator::printFileInfo(file1);
    BinaryFileComparator::printFileInfo(file2);
    
    // Compare files
    bool identical = BinaryFileComparator::compareFiles(file1, file2, tolerance, true);
    
    if (identical) {
        std::cout << "\nFiles are identical (within tolerance)" << std::endl;
        return 0;
    } else {
        std::cout << "\nFiles are different" << std::endl;
        return 1;
    }
}