import ProcessingStatus from '../ProcessingStatus';

export default function ProcessingStatusExample() {
  const handleComplete = () => {
    console.log('Processing completed');
  };

  return (
    <div className="max-w-2xl mx-auto p-4">
      <ProcessingStatus 
        isProcessing={true} 
        fileName="sample_video.mp4"
        onComplete={handleComplete}
      />
    </div>
  );
}