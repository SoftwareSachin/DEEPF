# Deepfake Detection Website Design Guidelines

## Design Approach
**Reference-Based Approach**: Drawing inspiration from professional security tools like GitHub's security dashboard and Google's SafeBrowsing, combined with clean upload interfaces like WeTransfer and Dropbox.

## Core Design Elements

### Color Palette
**Dark Mode Primary** (matching user's simple frontend requirement):
- Background: 220 15% 8% (Deep navy-black)
- Surface: 220 10% 12% (Darker cards/containers)
- Text Primary: 0 0% 95% (Near white)
- Text Secondary: 220 5% 70% (Muted gray)
- Accent Green (REAL): 142 76% 36% (Confident green for authentic content)
- Accent Red (FAKE): 0 84% 60% (Alert red for detected deepfakes)
- Warning Orange: 38 92% 50% (Processing/uncertain states)

### Typography
- **Primary Font**: Inter via Google Fonts CDN
- **Monospace**: JetBrains Mono for technical data (confidence scores, file details)
- **Hierarchy**: text-3xl for main headings, text-lg for upload areas, text-sm for metadata

### Layout System
**Tailwind Spacing**: Consistent use of units 4, 8, 12, and 16 (p-4, m-8, gap-12, etc.)
- Generous whitespace reflecting "simple frontend" requirement
- Center-focused layout with max-width containers
- Minimal sections to avoid clutter

### Component Library

#### Upload Interface
- Large drag-and-drop zone with subtle border and hover states
- File type icons (video/image) with clear labeling
- Progress bars during upload with percentage indicators
- Support badges for MP4, AVI, MOV, JPG, PNG formats

#### Analysis Display
- Clean card-based results with face detection thumbnails
- Confidence meters using progress bars (0-100%)
- REAL/FAKE status badges with appropriate color coding
- Frame-by-frame timeline for video analysis
- Expandable technical details section

#### Navigation & Status
- Minimal top navigation with logo and simple menu
- Real-time processing indicators with subtle animations
- Clear action buttons (Analyze, Reset, Download Report)

## Visual Hierarchy
1. **Upload Area**: Largest visual element when idle
2. **Results Section**: Prominent display with color-coded confidence scores
3. **Technical Details**: Collapsible secondary information
4. **Processing Status**: Contextual overlays during analysis

## Animations
Minimal and purposeful only:
- Subtle fade-ins for results
- Progress bar animations during processing
- Gentle hover states on interactive elements

## Images
No large hero image required. Focus on:
- Small technical icons for file types
- Thumbnail previews of uploaded content
- Face detection boundary visualizations
- Simple logo/branding element in header

The design emphasizes functional clarity over visual complexity, ensuring the powerful detection capabilities remain the focus while maintaining an approachable, professional interface.