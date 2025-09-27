import FrameAnalysis from '../FrameAnalysis';

export default function FrameAnalysisExample() {
  // Mock frame data for demonstration
  const mockFrames = [
    {
      id: 1,
      timestamp: 0.5,
      confidence: 0.89,
      isDeepfake: true,
      faces: [
        {
          id: 'face-1',
          bbox: { x: 25, y: 20, width: 30, height: 40 },
          confidence: 0.89,
          isDeepfake: true
        }
      ]
    },
    {
      id: 2,
      timestamp: 1.0,
      confidence: 0.94,
      isDeepfake: false,
      faces: [
        {
          id: 'face-1',
          bbox: { x: 28, y: 22, width: 28, height: 38 },
          confidence: 0.94,
          isDeepfake: false
        }
      ]
    },
    {
      id: 3,
      timestamp: 1.5,
      confidence: 0.76,
      isDeepfake: true,
      faces: [
        {
          id: 'face-1',
          bbox: { x: 30, y: 25, width: 26, height: 35 },
          confidence: 0.76,
          isDeepfake: true
        }
      ]
    }
  ];

  const handleClose = () => {
    console.log('Close frame analysis');
  };

  return (
    <div className="p-4">
      <FrameAnalysis 
        frames={mockFrames}
        onClose={handleClose}
      />
    </div>
  );
}