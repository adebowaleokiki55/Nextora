import { CircleMarker, MapContainer, TileLayer, Tooltip } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { RISK_STYLES, fmt } from "./format";

const CENTER = [6.53, 3.48];

function radiusFor(cases) {
  if (!cases) return 8;
  return Math.max(8, Math.min(26, Math.sqrt(cases) * 1.3));
}

/**
 * LGAs as circles coloured by risk level (sized by next-week forecast),
 * facilities as small dots (red when short of any tracked commodity).
 */
export default function RiskMap({ risk, facilities, shortageFacilityIds, selectedId, onSelect }) {
  return (
    <div className="relative h-[420px] overflow-hidden rounded-xl ring-1 ring-slate-200">
      <MapContainer center={CENTER} zoom={10} scrollWheelZoom className="h-full w-full">
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {facilities.map((facility) => {
          const short = shortageFacilityIds.has(facility.id);
          return (
            <CircleMarker
              key={`f-${facility.id}`}
              center={[facility.latitude, facility.longitude]}
              radius={short ? 5 : 3}
              pathOptions={{
                color: short ? "#b91c1c" : "#475569",
                fillColor: short ? "#ef4444" : "#94a3b8",
                fillOpacity: 0.9,
                weight: 1,
              }}
            >
              <Tooltip>
                <strong>{facility.name}</strong>
                <br />
                {short ? "Short of malaria commodities" : facility.has_laboratory ? "Laboratory" : "Health centre"}
                {facility.data_origin === "SYNTHETIC" && " · demo facility"}
              </Tooltip>
            </CircleMarker>
          );
        })}

        {risk.map((row) => {
          const style = RISK_STYLES[row.risk_level] || RISK_STYLES.UNKNOWN;
          const selected = row.location.id === selectedId;
          const nextWeek = row.forecasts?.[0];
          return (
            <CircleMarker
              key={`l-${row.location.id}`}
              center={[row.location.latitude, row.location.longitude]}
              radius={radiusFor(nextWeek?.predicted_cases)}
              pathOptions={{
                color: selected ? "#0f172a" : style.color,
                weight: selected ? 3 : 1.5,
                fillColor: style.color,
                fillOpacity: selected ? 0.55 : 0.35,
              }}
              eventHandlers={{ click: () => onSelect(row.location.id) }}
            >
              <Tooltip direction="top">
                <strong>{row.location.name}</strong>
                <br />
                Risk: {style.label}
                {row.risk_score !== null && ` (${fmt(row.risk_score)}/100)`}
                {nextWeek && (
                  <>
                    <br />
                    Next week: ~{fmt(nextWeek.predicted_cases)} cases
                  </>
                )}
              </Tooltip>
            </CircleMarker>
          );
        })}
      </MapContainer>

      <div className="pointer-events-none absolute bottom-3 left-3 z-[1000] rounded-lg bg-white/95 px-3 py-2 text-[11px] shadow ring-1 ring-slate-200">
        <div className="mb-1 font-semibold text-slate-700">Forecast risk</div>
        {["VERY_HIGH", "HIGH", "MODERATE", "LOW"].map((level) => (
          <div key={level} className="flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: RISK_STYLES[level].color }} />
            {RISK_STYLES[level].label}
          </div>
        ))}
        <div className="mt-1 flex items-center gap-1.5">
          <span className="inline-block h-2 w-2 rounded-full bg-red-500" /> Facility short of stock
        </div>
      </div>
    </div>
  );
}
