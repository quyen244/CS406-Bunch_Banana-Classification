# FEATURE ANALYSIS: Banana Bunch Classification Prediction Page

## 1. Executive Summary
- **Description:** Trang web cho phép người dùng tải lên hình ảnh buồng chuối để hệ thống phân loại và dự đoán tình trạng/loại chuối thông qua mô hình AI.
- **User Goal:** Nhận được kết quả dự đoán chính xác kèm theo biểu đồ xác suất trực quan sau khi tải và xử lý ảnh (cắt/chỉnh sửa).

## 2. Component Architecture (Atomic Design)
- **Atoms:** 
    - `Button`: Nút tải lên, nút dự đoán, nút reset.
    - `Icon`: Icon upload, icon lỗi, icon thành công.
    - `ProgressBar`: Hiển thị độ tin cậy (confidence score).
    - `FilePicker`: Thành phần chọn tệp ẩn.
- **Molecules:** 
    - `UploadDropzone`: Khu vực kéo thả tệp hỗ trợ hiệu ứng hover.
    - `ImagePreviewer`: Hiển thị ảnh đã chọn.
    - `CropModal`: Modal chứa công cụ cắt ảnh (react-easy-crop).
    - `ResultItem`: Hiển thị nhãn và phần trăm xác suất.
- **Organisms:** 
    - `PredictionForm`: Bao gồm Dropzone, Preview và các nút điều khiển.
    - `AnalysisDashboard`: Hiển thị kết quả chi tiết kèm biểu đồ xác suất (Recharts/Chart.js).

## 3. Data & State Management
- **Local State:** 
    - `selectedFile`: Tệp ảnh gốc.
    - `croppedImage`: Ảnh sau khi đã cắt (dạng Blob/Base64).
    - `predictionResult`: Lưu kết quả từ API (label, confidence).
    - `isAnalyzing`: Trạng thái loading.
    - `error`: Lưu thông báo lỗi.
- **Global State:** Không cần thiết cho tính năng đơn lẻ này (có thể dùng React Context nếu mở rộng).
- **Server State:** Gọi API dự đoán (POST method).
- **TypeScript Definitions:**
    ```typescript
    interface PredictionResponse {
      label: string;
      confidence: number;
      probabilities: Record<string, number>; // Cho biểu đồ
    }
    ```

## 4. Technical Logic & Edge Cases
- **Happy Path:** 
    1. Người dùng kéo thả/chọn ảnh.
    2. Cắt ảnh để tập trung vào buồng chuối.
    3. Nhấn "Dự đoán".
    4. Hiển thị kết quả và biểu đồ.
- **Loading State:** Sử dụng Skeleton hoặc Spinner chuyên nghiệp trong lúc chờ API.
- **Empty State:** Hiển thị hướng dẫn tải ảnh khi chưa có dữ liệu.
- **Error Handling:** 
    - **Tệp không phải ảnh:** Kiểm tra MIME type (image/*) ngay tại client. Nếu API trả lỗi 4xx, hiển thị thông báo "Tệp không hợp lệ".
    - **Ảnh quá nặng:** Kiểm tra dung lượng (>5MB). Gợi ý người dùng hoặc tự động nén ảnh (client-side compression) bằng `browser-image-compression`.
    - **API Fail:** Hiển thị thông báo "Hệ thống đang bận, vui lòng thử lại sau".

## 5. Performance & Accessibility (A11y)
- **Performance:** 
    - Nén ảnh trước khi upload để tiết kiệm băng thông.
    - Lazy load các thư viện biểu đồ nặng.
- **Accessibility:** 
    - Hỗ trợ phím tắt và Focus ring cho các nút.
    - ARIA labels cho các thành phần điều khiển việc cắt ảnh.
