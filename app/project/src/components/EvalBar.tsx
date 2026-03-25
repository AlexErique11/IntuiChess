// Simplified eval bar component without info tooltips

interface EvalBarProps {
  value: number;
  label: string;
}

export default function EvalBar({ value, label }: EvalBarProps) {
  // Value ranges from -10 to +10
  // 0 = equal (50% white, 50% black)
  // Positive = advantage for white
  // Negative = advantage for black
  
  // Clamp value between -10 and 10
  const clampedValue = Math.max(-10, Math.min(10, value));
  
  // Convert to percentage (0-100)
  // -10 = 0% white (100% black)
  // 0 = 50% white (50% black)
  // +10 = 100% white (0% black)
  const whitePercentage = ((clampedValue + 10) / 20) * 100;
  
  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative w-8 h-[512px] bg-slate-800 rounded-lg overflow-hidden shadow-lg border-2 border-slate-600">
        {/* White portion (from bottom) */}
        <div
          className="absolute bottom-0 left-0 right-0 bg-white transition-all duration-300"
          style={{ height: `${whitePercentage}%` }}
        />
        
        {/* Center line at 50% */}
        <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-slate-400 -translate-y-1/2" />
        
        {/* Value display */}
        <div className="absolute top-2 left-1/2 -translate-x-1/2 bg-slate-900/80 text-white text-xs font-bold px-1 py-0.5 rounded">
          {value >= 0 ? '+' : ''}{value.toFixed(1)}
        </div>
      </div>
      
      <div className="text-xs font-medium text-slate-600 text-center max-w-[80px]">
        {label}
      </div>
    </div>
  );
}
