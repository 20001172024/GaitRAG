import { CartesianGrid, Legend, Line, LineChart, Tooltip, XAxis, YAxis } from "recharts";
import { downsampleIndices, type Recording } from "../lib/csv";

const MAX_POINTS = 1000;
const WIDTH = 640;
const HEIGHT = 200;

function sum(rec: Recording, cols: string[], i: number): number {
  return cols.reduce((acc, c) => acc + (rec.data[c]?.[i] ?? 0), 0);
}

export function SignalPlots({ recording }: { recording: Recording }) {
  const t = recording.data["timestamp_ms"] ?? [];
  const idx = downsampleIndices(recording.rows, MAX_POINTS);
  const t0 = t[0] ?? 0;
  const rows = idx.map((i) => ({
    t: Number((((t[i] ?? 0) - t0) / 1000).toFixed(2)),
    left: Math.round(sum(recording, ["fsr_l_heel", "fsr_l_meta", "fsr_l_toe"], i)),
    right: Math.round(sum(recording, ["fsr_r_heel", "fsr_r_meta", "fsr_r_toe"], i)),
    acc: Number(
      Math.hypot(
        recording.data["imu1_ax"]?.[i] ?? 0,
        recording.data["imu1_ay"]?.[i] ?? 0,
        recording.data["imu1_az"]?.[i] ?? 0,
      ).toFixed(2),
    ),
  }));

  return (
    <section aria-labelledby="signals-heading" className="rounded-xl border bg-white p-5 shadow-sm">
      <h2 id="signals-heading" className="text-lg font-semibold">
        Sensor signals
      </h2>
      <p className="mt-1 text-sm text-slate-600">
        {recording.rows} samples ({((t[t.length - 1] - t0) / 1000).toFixed(1)} s), drawn at up to{" "}
        {MAX_POINTS} points.
      </p>
      <div className="mt-3 overflow-x-auto">
        <h3 className="text-sm font-medium text-slate-700">
          Foot load (sum of heel, meta, toe FSRs)
        </h3>
        <LineChart width={WIDTH} height={HEIGHT} data={rows} margin={{ left: 8, right: 8 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="t" unit=" s" type="number" domain={["dataMin", "dataMax"]} />
          <YAxis width={48} />
          <Tooltip />
          <Legend />
          <Line
            dataKey="left"
            name="Left foot"
            stroke="#0f766e"
            dot={false}
            isAnimationActive={false}
          />
          <Line
            dataKey="right"
            name="Right foot"
            stroke="#b45309"
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
        <h3 className="mt-4 text-sm font-medium text-slate-700">IMU 1 acceleration magnitude</h3>
        <LineChart width={WIDTH} height={HEIGHT} data={rows} margin={{ left: 8, right: 8 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="t" unit=" s" type="number" domain={["dataMin", "dataMax"]} />
          <YAxis width={48} unit=" m/s²" />
          <Tooltip />
          <Line dataKey="acc" name="|a|" stroke="#334155" dot={false} isAnimationActive={false} />
        </LineChart>
      </div>
    </section>
  );
}
