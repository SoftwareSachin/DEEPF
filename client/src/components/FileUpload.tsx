import { useState, useCallback } from "react";
import { Upload, Film, Image as ImageIcon, FileVideo, FileImage, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  isProcessing?: boolean;
  uploadProgress?: number;
}

export default function FileUpload({ onFileSelect, isProcessing = false, uploadProgress = 0 }: FileUploadProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      if (isValidFileType(file)) {
        setSelectedFile(file);
        onFileSelect(file);
      }
    }
  }, [onFileSelect]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (isValidFileType(file)) {
        setSelectedFile(file);
        onFileSelect(file);
      }
    }
  }, [onFileSelect]);

  const isValidFileType = (file: File) => {
    const validTypes = [
      'image/jpeg', 'image/png', 'image/jpg',
      'video/mp4', 'video/avi', 'video/mov', 'video/quicktime'
    ];
    return validTypes.includes(file.type);
  };

  const getFileIcon = (file: File) => {
    if (file.type.startsWith('video/')) {
      return <FileVideo className="w-8 h-8 text-muted-foreground" />;
    }
    return <FileImage className="w-8 h-8 text-muted-foreground" />;
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const clearFile = () => {
    setSelectedFile(null);
  };

  if (selectedFile && !isProcessing) {
    return (
      <Card className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium">Selected File</h3>
          <Button 
            variant="ghost" 
            size="icon"
            onClick={clearFile}
            data-testid="button-clear-file"
          >
            <X className="w-4 h-4" />
          </Button>
        </div>
        
        <div className="flex items-center gap-4 p-4 rounded-lg bg-card border">
          {getFileIcon(selectedFile)}
          <div className="flex-1 min-w-0">
            <p className="font-medium truncate" data-testid="text-filename">
              {selectedFile.name}
            </p>
            <p className="text-sm text-muted-foreground" data-testid="text-filesize">
              {formatFileSize(selectedFile.size)}
            </p>
          </div>
        </div>

        <Button 
          className="w-full mt-4" 
          onClick={() => console.log('Analyze file:', selectedFile.name)}
          data-testid="button-analyze"
        >
          Analyze for Deepfakes
        </Button>
      </Card>
    );
  }

  if (isProcessing) {
    return (
      <Card className="p-6">
        <div className="text-center">
          <div className="animate-pulse mb-4">
            <Upload className="w-12 h-12 text-muted-foreground mx-auto" />
          </div>
          <h3 className="text-lg font-medium mb-2">Uploading...</h3>
          <Progress value={uploadProgress} className="w-full" data-testid="progress-upload" />
          <p className="text-sm text-muted-foreground mt-2">{uploadProgress}% complete</p>
        </div>
      </Card>
    );
  }

  return (
    <Card className={`p-8 transition-all duration-200 hover-elevate ${
      isDragOver ? 'border-primary bg-primary/5' : 'border-dashed border-2'
    }`}>
      <div
        className="text-center"
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        data-testid="dropzone-area"
      >
        <div className="mb-4">
          <Upload className="w-16 h-16 text-muted-foreground mx-auto" />
        </div>
        
        <h3 className="text-2xl font-medium mb-2">
          Deepfake Detection
        </h3>
        
        <p className="text-muted-foreground mb-6 max-w-md mx-auto">
          Upload an image or video to analyze for AI-generated content. 
          Drag and drop files here, or click to browse.
        </p>

        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-6">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <ImageIcon className="w-4 h-4" />
            <span>JPG, PNG</span>
          </div>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Film className="w-4 h-4" />
            <span>MP4, AVI, MOV</span>
          </div>
        </div>

        <input
          type="file"
          accept="image/*,video/*"
          onChange={handleFileInput}
          className="hidden"
          id="file-upload"
          data-testid="input-file"
        />
        
        <Button asChild className="px-8">
          <label htmlFor="file-upload" className="cursor-pointer" data-testid="button-upload">
            Choose File
          </label>
        </Button>
      </div>
    </Card>
  );
}