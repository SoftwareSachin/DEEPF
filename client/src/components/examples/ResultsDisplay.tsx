import ResultsDisplay from '../ResultsDisplay';

export default function ResultsDisplayExample() {
  // Example with fake detection result
  const fakeResult = {
    isDeepfake: true,
    confidence: 0.89,
    fileName: "suspicious_video.mp4",
    fileType: 'video' as const,
    processingTime: 12.3,
    frameCount: 180,
    modelUsed: ["EfficientNet-B4", "Xception"]
  };

  // Example with real detection result  
  const realResult = {
    isDeepfake: false,
    confidence: 0.94,
    fileName: "authentic_photo.jpg",
    fileType: 'image' as const,
    processingTime: 3.7,
    modelUsed: ["EfficientNet-B4", "ResNet-50"]
  };

  const handleViewFrames = () => {
    console.log('View frame analysis clicked');
  };

  const handleDownloadReport = () => {
    console.log('Download report clicked');
  };

  return (
    <div className="max-w-2xl mx-auto p-4 space-y-6">
      <div>
        <h3 className="text-lg font-medium mb-4">Fake Detection Result</h3>
        <ResultsDisplay 
          result={fakeResult}
          onViewFrames={handleViewFrames}
          onDownloadReport={handleDownloadReport}
        />
      </div>
      
      <div>
        <h3 className="text-lg font-medium mb-4">Real Detection Result</h3>
        <ResultsDisplay 
          result={realResult}
          onViewFrames={handleViewFrames}
          onDownloadReport={handleDownloadReport}
        />
      </div>
    </div>
  );
}