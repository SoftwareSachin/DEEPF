import FileUpload from '../FileUpload';

export default function FileUploadExample() {
  const handleFileSelect = (file: File) => {
    console.log('File selected:', file.name);
  };

  return (
    <div className="max-w-2xl mx-auto p-4">
      <FileUpload onFileSelect={handleFileSelect} />
    </div>
  );
}