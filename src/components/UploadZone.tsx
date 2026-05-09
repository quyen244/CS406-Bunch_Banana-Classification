'use client';

import React, { useCallback, useState } from 'react';
import { Upload, FileImage, AlertCircle } from 'lucide-react';

interface UploadZoneProps {
  onFileSelect: (file: File) => void;
  error?: string;
}

const UploadZone: React.FC<UploadZoneProps> = ({ onFileSelect, error }) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      onFileSelect(files[0]);
    }
  }, [onFileSelect]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      onFileSelect(files[0]);
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`relative border-2 border-dashed rounded-2xl p-12 transition-all duration-300 flex flex-col items-center justify-center cursor-pointer
          ${isDragging 
            ? 'border-blue-500 bg-blue-50 scale-[1.02]' 
            : 'border-slate-300 hover:border-blue-400 hover:bg-slate-50'
          }`}
        onClick={() => document.getElementById('fileInput')?.click()}
      >
        <input
          id="fileInput"
          type="file"
          className="hidden"
          accept="image/*"
          onChange={handleFileChange}
        />
        
        <div className="bg-blue-100 p-4 rounded-full mb-4 group-hover:scale-110 transition-transform">
          <Upload className="w-8 h-8 text-blue-600" />
        </div>
        
        <h3 className="text-xl font-semibold text-slate-800 mb-2">
          Kéo thả hoặc nhấn để tải ảnh
        </h3>
        <p className="text-slate-500 text-center">
          Hỗ trợ JPG, PNG (Tối đa 5MB)
        </p>

        {error && (
          <div className="mt-4 flex items-center gap-2 text-red-500 bg-red-50 px-4 py-2 rounded-lg">
            <AlertCircle className="w-4 h-4" />
            <span className="text-sm font-medium">{error}</span>
          </div>
        )}
      </div>
      
      <div className="mt-8 grid grid-cols-3 gap-4 text-center">
        <div className="p-4 rounded-xl bg-white border border-slate-100 shadow-sm">
          <div className="text-blue-600 font-bold text-lg mb-1">1</div>
          <p className="text-xs text-slate-500">Tải ảnh lên</p>
        </div>
        <div className="p-4 rounded-xl bg-white border border-slate-100 shadow-sm">
          <div className="text-blue-600 font-bold text-lg mb-1">2</div>
          <p className="text-xs text-slate-500">Cắt & Chỉnh sửa</p>
        </div>
        <div className="p-4 rounded-xl bg-white border border-slate-100 shadow-sm">
          <div className="text-blue-600 font-bold text-lg mb-1">3</div>
          <p className="text-xs text-slate-500">Xem kết quả</p>
        </div>
      </div>
    </div>
  );
};

export default UploadZone;
