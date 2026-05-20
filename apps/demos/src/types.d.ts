// Ambient declarations for assets imported across the workspace.
declare module '*.wgsl?raw' {
  const src: string;
  export default src;
}
declare module '*.wgsl' {
  const src: string;
  export default src;
}
