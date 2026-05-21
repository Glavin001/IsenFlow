// Integrate water depth from flux fields and derive cell velocity (u, v).

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> water:          array<f32>;
@group(0) @binding(2) var<storage, read>       fluxLR:         array<f32>;
@group(0) @binding(3) var<storage, read>       fluxUD:         array<f32>;
@group(0) @binding(4) var<storage, read_write> velocity:       array<f32>;
@group(0) @binding(5) var<storage, read>       boundary:       array<u32>;
@group(0) @binding(6) var<storage, read>       boundaryTargetH: array<f32>;

fn lr_at(arr: ptr<storage, array<f32>, read>, p: vec2<i32>, w: i32, h: i32) -> vec2<f32> {
  if (!in_bounds(p, w, h)) { return vec2<f32>(0.0, 0.0); }
  let i = cell_idx(p.x, p.y, w);
  return vec2<f32>((*arr)[i * 2u + 0u], (*arr)[i * 2u + 1u]);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = i32(params.width);
  let h = i32(params.height);
  let p = vec2<i32>(i32(gid.x), i32(gid.y));
  if (!in_bounds(p, w, h)) { return; }
  let idx = cell_idx(p.x, p.y, w);

  let self_LR = lr_at(&fluxLR, p, w, h);
  let self_UD = lr_at(&fluxUD, p, w, h);

  let nbr_L_LR = lr_at(&fluxLR, p + vec2<i32>(-1, 0), w, h);
  let nbr_R_LR = lr_at(&fluxLR, p + vec2<i32>(1, 0),  w, h);
  let nbr_D_UD = lr_at(&fluxUD, p + vec2<i32>(0, -1), w, h);
  let nbr_U_UD = lr_at(&fluxUD, p + vec2<i32>(0, 1),  w, h);

  let outflow = self_LR.x + self_LR.y + self_UD.x + self_UD.y;
  let inflow  = nbr_L_LR.y + nbr_R_LR.x + nbr_D_UD.y + nbr_U_UD.x;

  let area = params.dx * params.dx;
  let dV   = (inflow - outflow) * params.dt;
  let dH   = dV / area;

  let h_old = water[idx * 2u];
  var new_h = max(0.0, h_old + dH);

  let bt = boundary[idx];
  if (bt == 4u) {
    // Inflow: pin to target depth (floor)
    new_h = max(new_h, boundaryTargetH[idx]);
  } else if (bt == 5u) {
    // Sea: pin to target depth
    new_h = boundaryTargetH[idx];
  } else if (bt == 3u) {
    let mean = (h_at(&water, p + vec2<i32>(-1, 0), w, h) +
                h_at(&water, p + vec2<i32>(1, 0),  w, h) +
                h_at(&water, p + vec2<i32>(0, -1), w, h) +
                h_at(&water, p + vec2<i32>(0, 1),  w, h)) * 0.25;
    new_h = mix(new_h, mean, 0.1);
  }

  water[idx * 2u + 0u] = new_h;
  water[idx * 2u + 1u] = h_old;

  // Mei 2007 eq. 8: velocity reconstruction — the four-flux sum is
  // explicitly divided by 2 (eq. 8 averages two face pairs).
  let net_x = (self_LR.y - self_LR.x + nbr_L_LR.y - nbr_R_LR.x) * 0.5;
  let net_z = (self_UD.y - self_UD.x + nbr_D_UD.y - nbr_U_UD.x) * 0.5;
  let h_avg = max(0.05, 0.5 * (h_old + new_h));
  let u = net_x / (params.dx * h_avg);
  let v = net_z / (params.dx * h_avg);
  velocity[idx * 2u + 0u] = u;
  velocity[idx * 2u + 1u] = v;
}
