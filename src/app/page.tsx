'use client';

import React, { useState, useCallback } from 'react';
import Image from 'next/image';
import imageCompression from 'browser-image-compression';
import { Sparkles, AlertCircle, Loader2, Cpu, Brain } from 'lucide-react';

import UploadZone from '@/components/UploadZone';
import ImageCropper from '@/components/ImageCropper';
import PredictionResult from '@/components/PredictionResult';

// ---------------------------------------------------------------------------
// Config: Gateway URL từ environment variable (hoặc fallback localhost)
// ---------------------------------------------------------------------------
const GATEWAY_URL =
  process.env.NEXT_PUBLIC_GATEWAY_URL ?? 'http://localhost:8080';

const DL_PREDICT_URL = `${GATEWAY_URL}/predict/dl`;
const ML_PREDICT_URL = `${GATEWAY_URL}/predict/ml`;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
type ServerType = 'dl' | 'ml';

interface MLModelOption {
  value: string;
  label: string;
  description: string;
}

const ML_MODEL_OPTIONS: MLModelOption[] = [
  { value: 'best_svm', label: 'SVM', description: 'Support Vector Machine' },
  { value: 'best_xgboost', label: 'XGBoost', description: 'Gradient Boosting' },
  { value: 'best_random_forest', label: 'Random Forest', description: 'Ensemble Trees' },
  { value: 'best_histgradient', label: 'HistGradient', description: 'Histogram-based GB' },
];

interface PredictionResponse {
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
  model?: string;
  server: ServerType;
  latency_ms: number;
}

interface FormattedResult {
  label: string;
  confidence: number;
  allProbabilities: { name: string; value: number }[];
  server: ServerType;
  model?: string;
  latency_ms: number;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export default function PredictionPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [croppedImage, setCroppedImage] = useState<Blob | null>(null);
  const [showCropper, setShowCropper] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<FormattedResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Server & model selection
  const [selectedServer, setSelectedServer] = useState<ServerType>('dl');
  const [selectedModel, setSelectedModel] = useState<string>('best_svm');

  // 1. Xử lý khi chọn tệp
  const handleFileSelect = useCallback(async (selectedFile: File) => {
    setError(null);
    setResult(null);

    if (!selectedFile.type.startsWith('image/')) {
      setError('Vui lòng chọn tệp tin hình ảnh hợp lệ.');
      return;
    }

    let fileToProcess = selectedFile;

    if (selectedFile.size > 5 * 1024 * 1024) {
      setError('Ảnh quá lớn (>5MB), đang tự động nén để tối ưu...');
      try {
        fileToProcess = await imageCompression(selectedFile, {
          maxSizeMB: 1,
          maxWidthOrHeight: 1920,
          useWebWorker: true,
        });
        setError(null);
      } catch {
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
    setPreview(URL.createObjectURL(croppedBlob));
  }, []);

  // 3. Gửi dự đoán lên Gateway
  const handlePredict = async () => {
    if (!croppedImage) return;

    setIsAnalyzing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', croppedImage, 'banana_bunch.jpg');

      // Xác định URL dựa theo server được chọn
      const url =
        selectedServer === 'dl'
          ? DL_PREDICT_URL
          : `${ML_PREDICT_URL}?model_name=${selectedModel}`;

      const response = await fetch(url, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        const detail = (errData as { detail?: string }).detail;
        if (response.status >= 400 && response.status < 500) {
          throw new Error(detail ?? 'Dữ liệu không hợp lệ. Vui lòng kiểm tra lại ảnh chụp.');
        }
        throw new Error(detail ?? 'Máy chủ gặp sự cố. Vui lòng thử lại sau.');
      }

      const data: PredictionResponse = await response.json();

      const formattedProbabilities = Object.entries(data.probabilities)
        .map(([name, value]) => ({ name, value }))
        .sort((a, b) => b.value - a.value);

      setResult({
        label: data.label,
        confidence: data.confidence,
        allProbabilities: formattedProbabilities,
        server: data.server,
        model: data.model,
        latency_ms: data.latency_ms,
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Đã xảy ra lỗi không xác định.';
      setError(message);
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
      {/* Header */}
      <div className="max-w-4xl mx-auto text-center mb-16">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-50 text-blue-600 text-sm font-bold mb-6">
          <Sparkles className="w-4 h-4" />
          Powered by CS406 AI Team
        </div>
        <h1 className="text-5xl font-black text-slate-900 tracking-tight mb-6">
          Phân loại <span className="text-blue-600">Buồng Chuối</span>
        </h1>
        <p className="text-xl text-slate-600 max-w-2xl mx-auto leading-relaxed">
          Tải lên hình ảnh buồng chuối để nhận phân tích chi tiết từ mô hình DL hoặc ML.
        </p>
      </div>

      {/* Main Content */}
      {!result ? (
        <div className="max-w-3xl mx-auto">
          {!preview ? (
            <>
              {/* Server Selection */}
              <div className="bg-white rounded-2xl p-6 border border-slate-100 shadow-sm mb-6">
                <p className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-4">
                  Chọn mô hình phân tích
                </p>
                <div className="grid grid-cols-2 gap-3">
                  {/* DL Option */}
                  <button
                    id="btn-server-dl"
                    onClick={() => setSelectedServer('dl')}
                    className={`flex items-center gap-3 p-4 rounded-xl border-2 transition-all text-left ${
                      selectedServer === 'dl'
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-slate-100 hover:border-slate-200'
                    }`}
                  >
                    <div className={`p-2 rounded-lg ${selectedServer === 'dl' ? 'bg-blue-500 text-white' : 'bg-slate-100 text-slate-500'}`}>
                      <Brain className="w-5 h-5" />
                    </div>
                    <div>
                      <p className="font-bold text-slate-800">Deep Learning</p>
                      <p className="text-xs text-slate-500">TensorFlow CNN</p>
                    </div>
                  </button>

                  {/* ML Option */}
                  <button
                    id="btn-server-ml"
                    onClick={() => setSelectedServer('ml')}
                    className={`flex items-center gap-3 p-4 rounded-xl border-2 transition-all text-left ${
                      selectedServer === 'ml'
                        ? 'border-violet-500 bg-violet-50'
                        : 'border-slate-100 hover:border-slate-200'
                    }`}
                  >
                    <div className={`p-2 rounded-lg ${selectedServer === 'ml' ? 'bg-violet-500 text-white' : 'bg-slate-100 text-slate-500'}`}>
                      <Cpu className="w-5 h-5" />
                    </div>
                    <div>
                      <p className="font-bold text-slate-800">Machine Learning</p>
                      <p className="text-xs text-slate-500">HOG + LBP + Sklearn</p>
                    </div>
                  </button>
                </div>

                {/* ML Model Selector */}
                {selectedServer === 'ml' && (
                  <div className="mt-4 pt-4 border-t border-slate-100">
                    <p className="text-sm font-semibold text-slate-500 mb-3">Chọn thuật toán ML</p>
                    <div className="grid grid-cols-2 gap-2">
                      {ML_MODEL_OPTIONS.map((opt) => (
                        <button
                          key={opt.value}
                          id={`btn-model-${opt.value}`}
                          onClick={() => setSelectedModel(opt.value)}
                          className={`p-3 rounded-lg border-2 text-left transition-all ${
                            selectedModel === opt.value
                              ? 'border-violet-500 bg-violet-50'
                              : 'border-slate-100 hover:border-slate-200'
                          }`}
                        >
                          <p className={`font-bold text-sm ${selectedModel === opt.value ? 'text-violet-700' : 'text-slate-700'}`}>
                            {opt.label}
                          </p>
                          <p className="text-xs text-slate-400">{opt.description}</p>
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <UploadZone onFileSelect={handleFileSelect} error={error ?? undefined} />
            </>
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

              {/* Selected config badge */}
              <div className="mb-6 flex items-center gap-2 text-sm">
                <span className={`px-3 py-1 rounded-full font-semibold ${
                  selectedServer === 'dl' ? 'bg-blue-100 text-blue-700' : 'bg-violet-100 text-violet-700'
                }`}>
                  {selectedServer === 'dl' ? '🧠 Deep Learning (TF CNN)' : `⚙️ ML — ${ML_MODEL_OPTIONS.find(m => m.value === selectedModel)?.label ?? selectedModel}`}
                </span>
              </div>

              {error && (
                <div className="mb-6 flex items-center gap-2 text-red-500 bg-red-50 px-6 py-3 rounded-xl w-full">
                  <AlertCircle className="w-5 h-5" />
                  <span className="font-medium">{error}</span>
                </div>
              )}

              <div className="flex gap-4 w-full max-w-md">
                <button
                  id="btn-cancel"
                  onClick={handleReset}
                  disabled={isAnalyzing}
                  className="flex-1 py-4 px-6 rounded-2xl border-2 border-slate-100 text-slate-500 font-bold hover:bg-slate-50 transition-all disabled:opacity-50"
                >
                  Hủy bỏ
                </button>
                <button
                  id="btn-predict"
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
        <>
          {/* Inference metadata badge */}
          <div className="max-w-3xl mx-auto mb-4 flex justify-center">
            <span className="text-xs px-3 py-1 rounded-full bg-slate-100 text-slate-500 font-mono">
              {result.server === 'dl'
                ? `DL Server (TF CNN) • ${result.latency_ms}ms`
                : `ML Server (${result.model ?? selectedModel}) • ${result.latency_ms}ms`}
            </span>
          </div>
          <PredictionResult result={result} onReset={handleReset} />
        </>
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
