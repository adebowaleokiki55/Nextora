import { useEffect, useRef } from "react";
import Plotly from "plotly.js-basic-dist-min";

const BASE_LAYOUT = {
  margin: { l: 44, r: 16, t: 8, b: 36 },
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { family: "DM Sans, system-ui, sans-serif", size: 11, color: "#475569" },
  legend: { orientation: "h", y: -0.25, x: 0, font: { size: 10 } },
  xaxis: { gridcolor: "#f1f5f9", linecolor: "#e2e8f0" },
  yaxis: { gridcolor: "#f1f5f9", linecolor: "#e2e8f0", rangemode: "tozero" },
  hovermode: "x unified",
};

/** Thin wrapper around Plotly (basic bundle: scatter + bar). */
export default function Plot({ data, layout, height = 260 }) {
  const ref = useRef(null);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    Plotly.react(
      element,
      data,
      { ...BASE_LAYOUT, ...layout, height },
      { displayModeBar: false, responsive: true }
    );
  }, [data, layout, height]);

  useEffect(() => {
    const element = ref.current;
    return () => {
      if (element) Plotly.purge(element);
    };
  }, []);

  return <div ref={ref} className="w-full" style={{ minHeight: height }} />;
}
