// Integrate water depth from flux fields and derive cell velocity (u, v).

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var waterTex:  texture_storage_2d<rg32float, read_write>;
@group(0) @binding(2) var fluxLR:    texture_storage_2d<rg32float, read>;
@group(0) @binding(3) var fluxUD:    texture_storage_2d<rg32float, read>;
@group(0) @binding(4) var velocity:  texture_storage_2d<rg32float, write>;
@group(0) @binding(5) var boundary:  texture_storage_2d<r32uint,   read>;

fn load_fLR(t: texture_storage_2d<rg32float, read>, p: vec2<i32>, w: i32, h: i32) -> vec2<f32> {
  if (p.x < 0 || p.y < 0 || p.x >= w || p.y >= h) { return vec2<f32>(0.0); }
  return textureLoad(t, p).xy;
}
fn load_fUD(t: texture_storage_2d<rg32float, read>, p: vec2<i32>, w: i32, h: i32) -> vec2<f32> {
  if (p.x < 0 || p.y < 0 || p.x >= w || p.y >= h) { return vec2<f32>(0.0); }
  return textureLoad(t, p).xy;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = i32(params.width);
  let h = i32(params.height);
  let p = vec2<i32>(i32(gid.x), i32(gid.y));
  if (!in_bounds(p, w, h)) { return; }

  let self_LR = load_fLR(fluxLR, p, w, h);
  let self_UD = load_fUD(fluxUD, p, w, h);

  // Inflow from neighbors = their outflow toward us.
  let nbr_L_LR = load_fLR(fluxLR, p + vec2<i32>(-1, 0), w, h); // its .y is outflow→right→our cell
  let nbr_R_LR = load_fLR(fluxLR, p + vec2<i32>(1, 0),  w, h); // its .x is outflow→left
  let nbr_D_UD = load_fUD(fluxUD, p + vec2<i32>(0, -1), w, h); // its .y is outflow→up
  let nbr_U_UD = load_fUD(fluxUD, p + vec2<i32>(0, 1),  w, h); // its .x is outflow→down

  let outflow = self_LR.x + self_LR.y + self_UD.x + self_UD.y;
  let inflow  = nbr_L_LR.y + nbr_R_LR.x + nbr_D_UD.y + nbr_U_UD.x;

  let area = params.dx * params.dx;
  let dV   = (inflow - outflow) * params.dt;
  let dH   = dV / area;

  var old = textureLoad(waterTex, p);
  var new_h = max(0.0, old.x + dH);

  let bt = textureLoad(boundary, p).x;
  if (bt == 4u) {
    // Inflow source: hold at last set depth (R channel) — old.x already includes user input.
    new_h = max(new_h, old.x);
  } else if (bt == 5u) {
    // Sea level clamp.
    new_h = old.x;   // assumed pre-set by user
  } else if (bt == 3u) {
    // Sponge: pull h toward neighborhood mean.
    let mean = (safe_load_height(waterTex, p + vec2<i32>(-1, 0), w, h) +
                safe_load_height(waterTex, p + vec2<i32>(1, 0),  w, h) +
                safe_load_height(waterTex, p + vec2<i32>(0, -1), w, h) +
                safe_load_height(waterTex, p + vec2<i32>(0, 1),  w, h)) * 0.25;
    new_h = mix(new_h, mean, 0.1);
  }

  textureStore(waterTex, p, vec4<f32>(new_h, old.x, 0.0, 0.0));

  // Derive cell velocity (u along +x, v along +z).
  let net_x = (self_LR.y - self_LR.x + nbr_L_LR.y - nbr_R_LR.x);
  let net_z = (self_UD.y - self_UD.x + nbr_D_UD.y - nbr_U_UD.x);
  let h_avg = max(0.05, 0.5 * (old.x + new_h));
  let u = net_x / (params.dx * h_avg);
  let v = net_z / (params.dx * h_avg);
  textureStore(velocity, p, vec4<f32>(u, v, 0.0, 0.0));
}
