'use client';

import React from 'react';

interface TimelineControllerProps {
  startTime: number;
  endTime: number;
  currentTime: number;
  isPlaying: boolean;
  playbackSpeed: number;
  onTimeChange: (time: number) => void;
  onPlayPause: (isPlaying: boolean) => void;
  onSpeedChange: (speed: number) => void;
}

const TimelineController: React.FC<TimelineControllerProps> = ({
  startTime,
  endTime,
  currentTime,
  isPlaying,
  playbackSpeed,
  onTimeChange,
  onPlayPause,
  onSpeedChange
}) => {
  const formatTime = (minutes: number): string => {
    const hours = Math.floor(minutes / 60);
    const mins = Math.floor(minutes % 60);
    return `${hours.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}`;
  };

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onTimeChange(parseFloat(e.target.value));
  };

  const handlePlayPause = () => {
    // Force a definitive true/false value rather than toggling
    const newPlayState = !isPlaying;
    console.log("TimelineController: Setting play state to:", newPlayState);
    onPlayPause(newPlayState);
  };

  const handleRestart = () => {
    onTimeChange(startTime);
    // Always start playing after reset
    onPlayPause(true);
  };

  const sliderPercentage = ((currentTime - startTime) / (endTime - startTime)) * 100;

  return (
    <div className="timeline-controller">
      <div className="timeline-info">
        <span className="time-display">
          {formatTime(currentTime)} / {formatTime(endTime)}
        </span>
      </div>
      
      <div className="timeline-slider-container">
        <input
          type="range"
          min={startTime}
          max={endTime}
          value={currentTime}
          onChange={handleSliderChange}
          className="timeline-slider"
          step="0.1"
        />
        <div 
          className="timeline-progress" 
          style={{ width: `${sliderPercentage}%` }}
        ></div>
      </div>
      
      <div className="timeline-controls">
        <button 
          className="control-button" 
          onClick={handlePlayPause}
          title={isPlaying ? "Pause" : "Play"}
        >
          {isPlaying ? '⏸️' : '▶️'}
        </button>
        
        <button 
          className="control-button"
          onClick={handleRestart}
          title="Restart animation"
        >
          ⏮️
        </button>
        
        <div className="speed-controls">
          <span>Speed:</span>
          {[1, 2, 5, 10].map(speed => (
            <button
              key={speed}
              className={`speed-button ${playbackSpeed === speed ? 'active' : ''}`}
              onClick={() => onSpeedChange(speed)}
              title={`${speed}x speed`}
            >
              {speed}x
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default TimelineController;