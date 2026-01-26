// frontend2\src\app\hooks\useDraggablePanel.tsx
import { useRef, useEffect, useCallback } from 'react';

interface DraggablePanelOptions {
  handleSelector: string;
  initialPosition?: { x: number, y: number };
  bounds?: {
    top: number;
    right: number;
    bottom: number;
    left: number;
  };
  onPositionChange?: (position: { x: number, y: number }) => void;
}

export function useDraggablePanel(
  elementRef: React.RefObject<HTMLElement>,
  options: DraggablePanelOptions
) {
  const {
    handleSelector,
    initialPosition,
    bounds = { top: 0, right: 0, bottom: 0, left: 0 },
    onPositionChange
  } = options;

  const isDraggingRef = useRef(false);
  const positionRef = useRef({ x: 0, y: 0 }); // Stores pixel position once hook takes over
  const hasPositionBeenSetByHookRef = useRef(false); // Tracks if style.left/top are set by hook

  const adjustToBounds = useCallback(() => {
    const element = elementRef.current;
    if (!element || !hasPositionBeenSetByHookRef.current) {
      // Only adjust if the hook is managing the pixel position
      return;
    }

    // Ensure we read the current pixel style if available, otherwise from rect
    let currentLeft = parseInt(element.style.left, 10);
    let currentTop = parseInt(element.style.top, 10);

    if (isNaN(currentLeft) || isNaN(currentTop)) { // Fallback if not set or NaN
        const rect = element.getBoundingClientRect();
        currentLeft = rect.left;
        currentTop = rect.top;
    }
    
    const elementWidth = element.offsetWidth;
    const elementHeight = element.offsetHeight;
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;

    const minX = bounds.left;
    const maxX = Math.max(minX, viewportWidth - elementWidth - bounds.right);
    const minY = bounds.top;
    const maxY = Math.max(minY, viewportHeight - elementHeight - bounds.bottom);

    const correctedX = Math.min(Math.max(currentLeft, minX), maxX);
    const correctedY = Math.min(Math.max(currentTop, minY), maxY);

    if (correctedX !== currentLeft || correctedY !== currentTop) {
      element.style.left = `${correctedX}px`;
      element.style.top = `${correctedY}px`;
      positionRef.current = { x: correctedX, y: correctedY };
      if (onPositionChange) {
        onPositionChange(positionRef.current);
      }
    }
  }, [elementRef, bounds, onPositionChange]);

  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;

    const handle = (element.querySelector(handleSelector) as HTMLElement) || element;
    if (handle) {
      handle.style.userSelect = 'none';
      handle.style.cursor = 'grab';
    }

    if (initialPosition) {
      element.style.left = `${initialPosition.x}px`;
      element.style.top = `${initialPosition.y}px`;
      positionRef.current = initialPosition;
      hasPositionBeenSetByHookRef.current = true;
      // If an initialPosition is provided, and the element also has CSS transform for centering,
      // it might be beneficial to also set element.style.transform = 'none'; here.
      // However, the primary issue reported is for CSS-only initial positioning.
      const timerId = setTimeout(() => adjustToBounds(), 50); // Adjust after setting
      return () => clearTimeout(timerId);
    } else {
      // Let CSS handle initial positioning.
      // Position will be captured and set on first drag.
      hasPositionBeenSetByHookRef.current = false;
    }

    let startX = 0, startY = 0, offsetX = 0, offsetY = 0;

    const onMouseDown = (e: MouseEvent) => {
      if ((e.target as HTMLElement)?.closest('button, input, select, textarea')) return;
      e.preventDefault();
      isDraggingRef.current = true;

      if (!hasPositionBeenSetByHookRef.current) {
        // First drag: capture current position (respecting CSS) and switch to pixel positioning
        const rect = element.getBoundingClientRect();
        offsetX = rect.left;
        offsetY = rect.top;
        element.style.left = `${offsetX}px`;
        element.style.top = `${offsetY}px`;
        element.style.transform = 'none'; // Reset transform to prevent jump
        hasPositionBeenSetByHookRef.current = true;
      } else {
        // Already using pixel positioning
        offsetX = parseInt(element.style.left, 10) || 0; // Fallback to 0 if style.left is not set/invalid
        offsetY = parseInt(element.style.top, 10) || 0;  // Fallback to 0 if style.top is not set/invalid
      }

      startX = e.clientX;
      startY = e.clientY;

      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseup', onMouseUp);
      element.classList.add('dragging');
      if (handle) handle.style.cursor = 'grabbing';
      document.body.style.userSelect = 'none';
    };

    const onMouseMove = (e: MouseEvent) => {
      if (!isDraggingRef.current) return;
      e.preventDefault();

      let newX = offsetX + (e.clientX - startX);
      let newY = offsetY + (e.clientY - startY);

      const elementWidth = element.offsetWidth;
      const elementHeight = element.offsetHeight;
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;

      const minX = bounds.left;
      const maxX = Math.max(minX, viewportWidth - elementWidth - bounds.right);
      const minY = bounds.top;
      const maxY = Math.max(minY, viewportHeight - elementHeight - bounds.bottom);

      newX = Math.min(Math.max(newX, minX), maxX);
      newY = Math.min(Math.max(newY, minY), maxY);

      element.style.left = `${newX}px`;
      element.style.top = `${newY}px`;
      positionRef.current = { x: newX, y: newY };
      if (onPositionChange) onPositionChange(positionRef.current);
    };

    const onMouseUp = () => {
      if (!isDraggingRef.current) return;
      isDraggingRef.current = false;
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
      element.classList.remove('dragging');
      if (handle) handle.style.cursor = 'grab';
      document.body.style.userSelect = '';
      if (hasPositionBeenSetByHookRef.current) { // Only adjust if hook is managing position
        adjustToBounds();
      }
    };
    
    // Touch events (similar logic to mouse events)
    const onTouchStart = (e: TouchEvent) => {
        if ((e.target as HTMLElement)?.closest('button, input, select, textarea')) return;
        if (e.touches.length !== 1) return;
        // e.preventDefault(); // Potentially handled by passive: false if needed

        isDraggingRef.current = true;
        if (!hasPositionBeenSetByHookRef.current) {
            const rect = element.getBoundingClientRect();
            offsetX = rect.left;
            offsetY = rect.top;
            element.style.left = `${offsetX}px`;
            element.style.top = `${offsetY}px`;
            element.style.transform = 'none'; // Reset transform to prevent jump
            hasPositionBeenSetByHookRef.current = true;
        } else {
            offsetX = parseInt(element.style.left, 10) || 0;
            offsetY = parseInt(element.style.top, 10) || 0;
        }
        startX = e.touches[0].clientX;
        startY = e.touches[0].clientY;

        document.addEventListener('touchmove', onTouchMove, { passive: false });
        document.addEventListener('touchend', onTouchEnd);
        element.classList.add('dragging');
        // document.body.style.userSelect = 'none'; // Consider for mobile
    };

    const onTouchMove = (e: TouchEvent) => {
        if (!isDraggingRef.current || e.touches.length !== 1) return;
        e.preventDefault(); // Prevent page scroll

        let newX = offsetX + (e.touches[0].clientX - startX);
        let newY = offsetY + (e.touches[0].clientY - startY);

        const elementWidth = element.offsetWidth;
        const elementHeight = element.offsetHeight;
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        const minX = bounds.left;
        const maxX = Math.max(minX, viewportWidth - elementWidth - bounds.right);
        const minY = bounds.top;
        const maxY = Math.max(minY, viewportHeight - elementHeight - bounds.bottom);
        newX = Math.min(Math.max(newX, minX), maxX);
        newY = Math.min(Math.max(newY, minY), maxY);

        element.style.left = `${newX}px`;
        element.style.top = `${newY}px`;
        positionRef.current = { x: newX, y: newY };
        if (onPositionChange) onPositionChange(positionRef.current);
    };

    const onTouchEnd = () => {
        if (!isDraggingRef.current) return;
        isDraggingRef.current = false;
        document.removeEventListener('touchmove', onTouchMove);
        document.removeEventListener('touchend', onTouchEnd);
        element.classList.remove('dragging');
        // document.body.style.userSelect = '';
        if (hasPositionBeenSetByHookRef.current) {
             adjustToBounds();
        }
    };

    const onWindowResize = () => {
      if (!isDraggingRef.current && hasPositionBeenSetByHookRef.current) {
        adjustToBounds();
      }
      // If CSS is handling position (hasPositionBeenSetByHookRef is false),
      // CSS will naturally re-center it on resize.
    };

    if (handle) {
      handle.addEventListener('mousedown', onMouseDown);
      handle.addEventListener('touchstart', onTouchStart, { passive: false });
    }

    window.addEventListener('resize', onWindowResize);

    return () => {
      handle?.removeEventListener('mousedown', onMouseDown);
      handle?.removeEventListener('touchstart', onTouchStart);
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
      document.removeEventListener('touchmove', onTouchMove);
      document.removeEventListener('touchend', onTouchEnd);
      window.removeEventListener('resize', onWindowResize);
    };
  }, [elementRef, handleSelector, initialPosition, bounds, onPositionChange, adjustToBounds]);

  return {
    isDragging: isDraggingRef.current,
    position: positionRef.current
  };
}