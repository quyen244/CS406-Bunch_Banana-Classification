'use client';

import React, { useState, useCallback } from 'react';
import Image from 'next/image';
import imageCompression from 'browser-image-compression';
import { Sparkles, AlertCircle, Loader2 } from 'lucide-react';

import UploadZone from '@/components/UploadZone';
import ImageCropper from '@/components/ImageCropper';
import PredictionResult from '@/components/PredictionResult';

// --- MOCK API LOGIC (Thay thế bằng URL API thật của bạn) ---
const PREDICT_API_URL = 'http://127.0.0.1:8000/predict'; // Cập nhật URL API của bạn ở đây

interface PredictionResponse {
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export default function PredictionPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [croppedImage, setCroppedImage] = useState<Blob | null>(null);
  const [showCropper, setShowCropper] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);

  // 1. Xử lý khi chọn tệp
  const handleFileSelect = useCallback(async (selectedFile: File) => {
    setError(null);
    setResult(null);

    // Kiểm tra định dạng ảnh
    if (!selectedFile.type.startsWith('image/')) {
      setError('Vui lòng chọn tệp tin hình ảnh hợp lệ.');
      return;
    }

    let fileToProcess = selectedFile;

    // Kiểm tra dung lượng (Nếu > 5MB thì yêu cầu nén)
    if (selectedFile.size > 5 * 1024 * 1024) {
      setError('Ảnh quá lớn (>5MB), đang tự động nén để tối ưu...');
      try {
        const options = {
          maxSizeMB: 1,
          maxWidthOrHeight: 1920,
          useWebWorker: true,
        };
        fileToProcess = await imageCompression(selectedFile, options);
        setError(null);
      } catch (err) {
        setError('Không thể nén ảnh. Vui lòng thử lại với ảnh nhỏ hơn.');
        return;
      }
    }

    setFile(fileToProcess);
    const reader = new FileReader();
    reader.onload = () => {
      setPreview(reader.result as string);
      setShowCropper(true);
    };
    reader.readAsDataURL(fileToProcess);
  }, []);

  // 2. Xử lý sau khi cắt ảnh
  const handleCropComplete = useCallback((croppedBlob: Blob) => {
    setCroppedImage(croppedBlob);
    setShowCropper(false);
    // Tự động tạo preview mới từ ảnh đã cắt
    setPreview(URL.createObjectURL(croppedBlob));
  }, []);

  // 3. Gửi dự đoán lên API
  const handlePredict = async () => {
    if (!croppedImage) return;

    setIsAnalyzing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', croppedImage, 'banana_bunch.jpg');

      const response = await fetch(PREDICT_API_URL, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        if (response.status >= 400 && response.status < 500) {
          throw new Error('Dữ liệu không hợp lệ. Vui lòng kiểm tra lại ảnh chụp.');
        }
        throw new Error('Máy chủ gặp sự cố. Vui lòng thử lại sau.');
      }

      const data: PredictionResponse = await response.json();
      
      // Chuyển đổi dữ liệu probabilities sang format của Recharts
      const formattedProbabilities = Object.entries(data.probabilities).map(([name, value]) => ({
        name,
        value,
      })).sort((a, b) => b.value - a.value);

      setResult({
        label: data.label,
        confidence: data.confidence,
        allProbabilities: formattedProbabilities,
      });
    } catch (err: any) {
      setError(err.message || 'Đã xảy ra lỗi không xác định.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setCroppedImage(null);
    setResult(null);
    setError(null);
  };

  return (
    <main className="min-h-screen bg-slate-50 py-12 px-4 sm:px-6 lg:px-8">
      {/* Header Section */}
      <div className="max-w-4xl mx-auto text-center mb-16">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-50 text-blue-600 text-sm font-bold mb-6">
          <Sparkles className="w-4 h-4" />
          Powered by CS406 AI Team
        </div>
        <h1 className="text-5xl font-black text-slate-900 tracking-tight mb-6">
          Phân loại <span className="text-blue-600">Buồng Chuối</span>
        </h1>
        <p className="text-xl text-slate-600 max-w-2xl mx-auto leading-relaxed">
          Tải lên hình ảnh buồng chuối của bạn để nhận phân tích chi tiết về loại và tình trạng chỉ trong vài giây.
        </p>
      </div>

      {/* Main Content Area */}
      {!result ? (
        <div className="max-w-3xl mx-auto">
          {!preview ? (
            <UploadZone onFileSelect={handleFileSelect} error={error || undefined} />
          ) : (
            <div className="bg-white rounded-3xl p-8 border border-slate-100 shadow-xl shadow-slate-200/50 flex flex-col items-center">
              <div className="relative w-full aspect-square max-w-md rounded-2xl overflow-hidden mb-8 border-4 border-slate-50">
                <Image 
                  src={preview} 
                  alt="Banana preview" 
                  fill 
                  className="object-cover"
                />
                <button 
                  onClick={() => setShowCropper(true)}
                  className="absolute bottom-4 right-4 bg-white/90 backdrop-blur px-4 py-2 rounded-lg text-sm font-bold text-slate-800 hover:bg-white transition-colors"
                >
                  Chỉnh sửa lại
                </button>
              </div>

              {error && (
                <div className="mb-6 flex items-center gap-2 text-red-500 bg-red-50 px-6 py-3 rounded-xl w-full">
                  <AlertCircle className="w-5 h-5" />
                  <span className="font-medium">{error}</span>
                </div>
              )}

              <div className="flex gap-4 w-full max-w-md">
                <button
                  onClick={handleReset}
                  disabled={isAnalyzing}
                  className="flex-1 py-4 px-6 rounded-2xl border-2 border-slate-100 text-slate-500 font-bold hover:bg-slate-50 transition-all disabled:opacity-50"
                >
                  Hủy bỏ
                </button>
                <button
                  onClick={handlePredict}
                  disabled={isAnalyzing}
                  className="flex-[2] py-4 px-6 rounded-2xl bg-blue-600 text-white font-bold hover:bg-blue-700 shadow-lg shadow-blue-200 transition-all flex items-center justify-center gap-3 disabled:opacity-70 disabled:cursor-not-allowed"
                >
                  {isAnalyzing ? (
                    <>
                      <Loader2 className="w-6 h-6 animate-spin" />
                      Đang phân tích...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-6 h-6" />
                      Dự đoán ngay
                    </>
                  )}
                </button>
              </div>
            </div>
          )}
        </div>
      ) : (
        <PredictionResult result={result} onReset={handleReset} />
      )}

      {/* Cropper Modal */}
      {showCropper && preview && (
        <ImageCropper 
          image={preview} 
          onCropComplete={handleCropComplete} 
          onCancel={() => {
            if (!croppedImage) {
              handleReset();
            } else {
              setShowCropper(false);
            }
          }}
        />
      )}

      {/* Footer */}
      <footer className="mt-24 text-center text-slate-400 text-sm">
        &copy; 2026 Banana Classification System. Built with &hearts; by CS406 Team.
      </footer>
    </main>
  );
}
