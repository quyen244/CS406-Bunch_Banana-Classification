'use client';

import React from 'react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Cell
} from 'recharts';
import { Trophy, Info, RefreshCcw } from 'lucide-react';

interface PredictionData {
  label: string;
  confidence: number;
  allProbabilities: { name: string; value: number }[];
}

interface PredictionResultProps {
  result: PredictionData;
  onReset: () => void;
}

const COLORS = ['#2563eb', '#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe'];

const PredictionResult: React.FC<PredictionResultProps> = ({ result, onReset }) => {
  return (
    <div className="w-full max-w-4xl mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <div className="grid md:grid-cols-2 gap-8">
        {/* Main Result Card */}
        <div className="bg-white rounded-3xl p-8 border border-slate-100 shadow-xl shadow-slate-200/50 flex flex-col items-center text-center">
          <div className="w-20 h-20 bg-yellow-100 rounded-full flex items-center justify-center mb-6">
            <Trophy className="w-10 h-10 text-yellow-600" />
          </div>
          <h2 className="text-sm font-bold text-blue-600 uppercase tracking-widest mb-2">Kết quả dự đoán</h2>
          <div className="text-4xl font-black text-slate-900 mb-4">{result.label}</div>
          <div className="flex flex-col items-center gap-1">
            <div className="text-5xl font-black text-blue-600">
              {(result.confidence * 100).toFixed(1)}%
            </div>
            <p className="text-slate-500 font-medium">Độ tin cậy</p>
          </div>
          
          <button
            onClick={onReset}
            className="mt-8 flex items-center gap-2 text-slate-500 hover:text-blue-600 font-semibold transition-colors"
          >
            <RefreshCcw className="w-4 h-4" />
            Thử lại với ảnh khác
          </button>
        </div>

        {/* Info Card */}
        <div className="bg-slate-900 rounded-3xl p-8 text-white flex flex-col justify-center">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-white/10 rounded-lg">
              <Info className="w-5 h-5 text-blue-400" />
            </div>
            <h3 className="text-lg font-bold">Thông tin chi tiết</h3>
          </div>
          <p className="text-slate-300 leading-relaxed mb-6">
            Dựa trên mô hình AI đã được huấn luyện, buồng chuối này được phân loại là <span className="text-white font-bold">{result.label}</span>. 
            Xác suất này phản ánh mức độ khớp của đặc điểm hình ảnh với dữ liệu mẫu trong hệ thống.
          </p>
          <div className="space-y-3">
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">Thời gian xử lý</span>
              <span className="font-mono text-blue-400">~240ms</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">Phiên bản mô hình</span>
              <span className="font-mono text-blue-400">v1.2.0</span>
            </div>
          </div>
        </div>
      </div>

      {/* Chart Section */}
      <div className="bg-white rounded-3xl p-8 border border-slate-100 shadow-xl shadow-slate-200/50">
        <h3 className="text-xl font-bold text-slate-800 mb-8 flex items-center gap-2">
          Biểu đồ xác suất các loại chuối
        </h3>
        <div className="h-[300px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={result.allProbabilities}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
              <XAxis type="number" hide />
              <YAxis 
                dataKey="name" 
                type="category" 
                tick={{ fill: '#64748b', fontSize: 12, fontWeight: 600 }}
                width={100}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip 
                cursor={{ fill: '#f8fafc' }}
                contentStyle={{ 
                  borderRadius: '12px', 
                  border: 'none', 
                  boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)',
                  fontSize: '13px'
                }}
                formatter={(value: any) => [`${(Number(value) * 100).toFixed(2)}%`, 'Xác suất']}
              />
              <Bar 
                dataKey="value" 
                radius={[0, 4, 4, 0]} 
                barSize={32}
              >
                {result.allProbabilities.map((_, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default PredictionResult;
