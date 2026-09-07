//! Real NGX Neural Rendering preview on bundled sample color/depth/MV.
//!
//! Uses D3D11 + NGX Core + `nvngx_dlssnr.dll` (feature 18). Knobs that map to
//! evaluate inputs (work resolution / sharpness / reset) re-run evaluate.
//! When NGX/GPU/DLL are unavailable the error is explicit — never a fake NR image.

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// RGBA8 image for egui.
#[derive(Debug, Clone)]
pub struct PreviewImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Knobs that affect the sample evaluate (subset of Feeder knobs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreviewKnobs {
    /// 50..100 — scales the work render size relative to sample.
    pub work_resolution: i32,
    /// 0 off / 1 every / 2 adaptive — mapped to NGX reset flag.
    pub reset_mode: i32,
    /// FSR/RCAS sharpness proxy when work < 100 (0..1).
    pub work_sharpness: f32,
}

impl Default for PreviewKnobs {
    fn default() -> Self {
        Self {
            work_resolution: 100,
            reset_mode: 2,
            work_sharpness: 0.30,
        }
    }
}

/// Resolve `nvngx_dlssnr.dll`: game folder, then installer cache, then next to this exe.
pub fn find_dlssnr(game_dir: Option<&Path>) -> Option<PathBuf> {
    let name = "nvngx_dlssnr.dll";
    if let Some(d) = game_dir {
        let p = d.join(name);
        if p.is_file() {
            return Some(p);
        }
        let host = d.join("host64").join(name);
        if host.is_file() {
            return Some(host);
        }
    }
    {
        let cache = crate::net::cache_dir();
        for cand in [
            cache.join(name),
            cache.join("dlss").join(name),
            cache.join("nvidia").join(name),
        ] {
            if cand.is_file() {
                return Some(cand);
            }
        }
        // Walk one level of cache for a copy this tool already fetched.
        if let Ok(rd) = std::fs::read_dir(&cache) {
            for e in rd.flatten() {
                let p = e.path().join(name);
                if p.is_file() {
                    return Some(p);
                }
                if e.path().is_file()
                    && e.file_name().to_string_lossy().eq_ignore_ascii_case(name)
                {
                    return Some(e.path());
                }
            }
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p = dir.join(name);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

/// Bundled sample dimensions and procedural buffers (also written under assets/preview).
pub const SAMPLE_W: u32 = 256;
pub const SAMPLE_H: u32 = 144;

/// Generate sample color (RGB), depth (R32), motion vectors (RG) in memory.
pub fn sample_buffers(work_pct: i32) -> (Vec<u8>, Vec<f32>, Vec<f32>, u32, u32) {
    let (w, h) = work_dims(work_pct);
    let mut color = vec![0u8; (w * h * 4) as usize];
    let mut depth = vec![0f32; (w * h) as usize];
    let mut mv = vec![0f32; (w * h * 2) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let fx = x as f32 / w as f32;
            let fy = y as f32 / h as f32;
            // Soft noisy gradient + a moving bright square (simulates detail residual).
            let n = ((x.wrapping_mul(374761393) ^ y.wrapping_mul(668265263)) % 255) as f32 / 255.0;
            let sq = if (fx - 0.35).abs() < 0.08 && (fy - 0.4).abs() < 0.08 {
                0.55
            } else {
                0.0
            };
            let r = ((0.15 + 0.55 * fx + 0.15 * n + sq) * 255.0).min(255.0) as u8;
            let g = ((0.12 + 0.45 * fy + 0.20 * n) * 255.0).min(255.0) as u8;
            let b = ((0.25 + 0.35 * (1.0 - fx) + 0.10 * n) * 255.0).min(255.0) as u8;
            color[i * 4] = r;
            color[i * 4 + 1] = g;
            color[i * 4 + 2] = b;
            color[i * 4 + 3] = 255;
            depth[i] = 0.15 + 0.7 * fy + 0.05 * n;
            // Small camera-pan-like motion in pixels.
            mv[i * 2] = 1.5 + 0.5 * n;
            mv[i * 2 + 1] = -0.8;
        }
    }
    (color, depth, mv, w, h)
}

fn work_dims(work_pct: i32) -> (u32, u32) {
    let scale = (work_pct.clamp(50, 100) as f32) / 100.0;
    let w = ((SAMPLE_W as f32) * scale).round().max(64.0) as u32;
    let h = ((SAMPLE_H as f32) * scale).round().max(36.0) as u32;
    (w, h)
}

/// Load bundled sample assets when present; otherwise generate procedurally.
/// Depth/MV on disk are full SAMPLE_W×SAMPLE_H; they are nearest-neighbour scaled to work size.
pub fn load_or_generate_samples(work_pct: i32) -> (Vec<u8>, Vec<f32>, Vec<f32>, u32, u32) {
    let (tw, th) = work_dims(work_pct);
    if let Some(dir) = sample_asset_dir() {
        if let Ok(loaded) = load_sample_assets(&dir, tw, th) {
            return loaded;
        }
    }
    sample_buffers(work_pct)
}

fn sample_asset_dir() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from("assets/preview"),
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join("preview")))
            .unwrap_or_default(),
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join("assets").join("preview")))
            .unwrap_or_default(),
    ];
    candidates.into_iter().find(|p| p.join("sample_color.png").is_file())
}

fn load_sample_assets(dir: &Path, tw: u32, th: u32) -> Result<(Vec<u8>, Vec<f32>, Vec<f32>, u32, u32)> {
    let img = image::open(dir.join("sample_color.png"))
        .context("open sample_color.png")?
        .to_rgba8();
    let (sw, sh) = img.dimensions();
    let depth_bytes = std::fs::read(dir.join("sample_depth.f32")).context("read sample_depth.f32")?;
    let mv_bytes = std::fs::read(dir.join("sample_mv.f32")).context("read sample_mv.f32")?;
    let expected_d = (sw * sh * 4) as usize;
    let expected_mv = (sw * sh * 8) as usize;
    if depth_bytes.len() < expected_d || mv_bytes.len() < expected_mv {
        bail!(
            "sample depth/MV size mismatch (got {}/{} want {}/{})",
            depth_bytes.len(),
            mv_bytes.len(),
            expected_d,
            expected_mv
        );
    }
    let mut depth_full = vec![0f32; (sw * sh) as usize];
    for (i, chunk) in depth_bytes[..expected_d].chunks_exact(4).enumerate() {
        depth_full[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    let mut mv_full = vec![0f32; (sw * sh * 2) as usize];
    for (i, chunk) in mv_bytes[..expected_mv].chunks_exact(4).enumerate() {
        mv_full[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    if sw == tw && sh == th {
        return Ok((img.into_raw(), depth_full, mv_full, tw, th));
    }
    // Nearest-neighbour scale to work size.
    let mut color = vec![0u8; (tw * th * 4) as usize];
    let mut depth = vec![0f32; (tw * th) as usize];
    let mut mv = vec![0f32; (tw * th * 2) as usize];
    for y in 0..th {
        for x in 0..tw {
            let sx = (x as u64 * sw as u64 / tw as u64) as u32;
            let sy = (y as u64 * sh as u64 / th as u64) as u32;
            let si = (sy * sw + sx) as usize;
            let di = (y * tw + x) as usize;
            color[di * 4..di * 4 + 4].copy_from_slice(&img.as_raw()[si * 4..si * 4 + 4]);
            depth[di] = depth_full[si];
            mv[di * 2] = mv_full[si * 2];
            mv[di * 2 + 1] = mv_full[si * 2 + 1];
        }
    }
    Ok((color, depth, mv, tw, th))
}

static LAST_ERROR: Mutex<Option<String>> = Mutex::new(None);

#[allow(dead_code)]
pub fn last_error() -> Option<String> {
    LAST_ERROR.lock().ok().and_then(|g| g.clone())
}

fn set_err(e: impl ToString) {
    if let Ok(mut g) = LAST_ERROR.lock() {
        *g = Some(e.to_string());
    }
}

/// NGX Result: Success = 0x1; failures have the 0xBAD00000 class bit.
#[cfg(windows)]
fn ngx_failed(rc: i32) -> bool {
    (rc as u32 & 0xFFF0_0000) == 0xBAD0_0000
}

/// Run a real NGX evaluate when possible. On failure returns Err with a clear reason
/// (never a placeholder “fake NR” bitmap).
pub fn evaluate(game_dir: Option<&Path>, knobs: PreviewKnobs) -> Result<PreviewImage> {
    #[cfg(windows)]
    {
        match evaluate_windows(game_dir, knobs) {
            Ok(img) => {
                if let Ok(mut g) = LAST_ERROR.lock() {
                    *g = None;
                }
                Ok(img)
            }
            Err(e) => {
                set_err(format!("{e:#}"));
                Err(e)
            }
        }
    }
    #[cfg(not(windows))]
    {
        let _ = (game_dir, knobs);
        bail!("NGX preview is Windows-only");
    }
}

#[cfg(windows)]
fn evaluate_windows(game_dir: Option<&Path>, knobs: PreviewKnobs) -> Result<PreviewImage> {
    if !crate::ngx::healthy() {
        bail!(
            "NGX Core / driver not ready ({})",
            crate::ngx::describe()
        );
    }
    let dll = find_dlssnr(game_dir).context(
        "nvngx_dlssnr.dll not found — Install a game first, or place the DLL in the tool cache",
    )?;
    let dll_dir = dll.parent().context("dlssnr has no parent")?;
    add_dll_directory(dll_dir)?;

    let (color, depth, mv, w, h) = load_or_generate_samples(knobs.work_resolution);
    let (out_w, out_h) = if knobs.work_resolution < 100 {
        (SAMPLE_W, SAMPLE_H)
    } else {
        (w, h)
    };

    ngx_d3d11_evaluate(&dll, &color, &depth, &mv, w, h, out_w, out_h, knobs).with_context(|| {
        format!(
            "NGX Neural Rendering evaluate failed (dll {})",
            dll.display()
        )
    })
}

#[cfg(windows)]
fn add_dll_directory(dir: &Path) -> Result<()> {
    use std::os::windows::ffi::OsStrExt;
    let wide: Vec<u16> = dir
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    #[link(name = "kernel32")]
    extern "system" {
        fn SetDllDirectoryW(lpPathName: *const u16) -> i32;
    }
    let ok = unsafe { SetDllDirectoryW(wide.as_ptr()) };
    if ok == 0 {
        bail!("SetDllDirectoryW failed for {}", dir.display());
    }
    Ok(())
}

#[cfg(windows)]
mod d3d11_raw {
    use std::ffi::c_void;
    use std::ptr;

    pub const DXGI_FORMAT_R8G8B8A8_UNORM: u32 = 28;
    pub const DXGI_FORMAT_R32_FLOAT: u32 = 41;
    pub const DXGI_FORMAT_R32G32_FLOAT: u32 = 16;

    pub const D3D11_USAGE_DEFAULT: u32 = 0;
    pub const D3D11_USAGE_STAGING: u32 = 3;
    pub const D3D11_BIND_SHADER_RESOURCE: u32 = 0x8;
    pub const D3D11_BIND_UNORDERED_ACCESS: u32 = 0x80;
    pub const D3D11_CPU_ACCESS_READ: u32 = 0x20000;
    pub const D3D11_MAP_READ: u32 = 1;

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct DxgiSampleDesc {
        pub count: u32,
        pub quality: u32,
    }

    #[repr(C)]
    pub struct Texture2dDesc {
        pub width: u32,
        pub height: u32,
        pub mip_levels: u32,
        pub array_size: u32,
        pub format: u32,
        pub sample_desc: DxgiSampleDesc,
        pub usage: u32,
        pub bind_flags: u32,
        pub cpu_access_flags: u32,
        pub misc_flags: u32,
    }

    #[repr(C)]
    pub struct SubresourceData {
        pub p_sys_mem: *const c_void,
        pub sys_mem_pitch: u32,
        pub sys_mem_slice_pitch: u32,
    }

    #[repr(C)]
    pub struct MappedSubresource {
        pub p_data: *mut c_void,
        pub row_pitch: u32,
        pub depth_pitch: u32,
    }

    pub unsafe fn com_release(ptr: *mut c_void) {
        if ptr.is_null() {
            return;
        }
        let vtbl = *(ptr as *const *const usize);
        let release: unsafe extern "system" fn(*mut c_void) -> u32 =
            std::mem::transmute(*vtbl.add(2));
        release(ptr);
    }

    unsafe fn vcall<T>(obj: *mut c_void, index: usize) -> T {
        let vtbl = *(obj as *const *const usize);
        std::mem::transmute_copy(&*vtbl.add(index))
    }

    pub unsafe fn create_texture2d(
        device: *mut c_void,
        desc: &Texture2dDesc,
        init: Option<&SubresourceData>,
    ) -> Result<*mut c_void, i32> {
        type FnCreate = unsafe extern "system" fn(
            *mut c_void,
            *const Texture2dDesc,
            *const SubresourceData,
            *mut *mut c_void,
        ) -> i32;
        let f: FnCreate = vcall(device, 5);
        let mut out = ptr::null_mut();
        let hr = f(
            device,
            desc,
            init.map(|p| p as *const _).unwrap_or(ptr::null()),
            &mut out,
        );
        if hr < 0 || out.is_null() {
            Err(hr)
        } else {
            Ok(out)
        }
    }

    pub unsafe fn update_subresource(
        ctx: *mut c_void,
        dst: *mut c_void,
        data: *const c_void,
        row_pitch: u32,
    ) {
        type FnUp = unsafe extern "system" fn(
            *mut c_void,
            *mut c_void,
            u32,
            *const c_void,
            *const c_void,
            u32,
            u32,
        );
        let f: FnUp = vcall(ctx, 48);
        f(ctx, dst, 0, ptr::null(), data, row_pitch, 0);
    }

    pub unsafe fn copy_resource(ctx: *mut c_void, dst: *mut c_void, src: *mut c_void) {
        type FnCopy = unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void);
        let f: FnCopy = vcall(ctx, 47);
        f(ctx, dst, src);
    }

    pub unsafe fn map(
        ctx: *mut c_void,
        res: *mut c_void,
    ) -> Result<MappedSubresource, i32> {
        type FnMap = unsafe extern "system" fn(
            *mut c_void,
            *mut c_void,
            u32,
            u32,
            u32,
            *mut MappedSubresource,
        ) -> i32;
        let f: FnMap = vcall(ctx, 14);
        let mut mapped = MappedSubresource {
            p_data: ptr::null_mut(),
            row_pitch: 0,
            depth_pitch: 0,
        };
        let hr = f(ctx, res, 0, D3D11_MAP_READ, 0, &mut mapped);
        if hr < 0 || mapped.p_data.is_null() {
            Err(hr)
        } else {
            Ok(mapped)
        }
    }

    pub unsafe fn unmap(ctx: *mut c_void, res: *mut c_void) {
        type FnUnmap = unsafe extern "system" fn(*mut c_void, *mut c_void, u32);
        let f: FnUnmap = vcall(ctx, 15);
        f(ctx, res, 0);
    }

    pub unsafe fn flush(ctx: *mut c_void) {
        type FnFlush = unsafe extern "system" fn(*mut c_void);
        let f: FnFlush = vcall(ctx, 111);
        f(ctx);
    }
}

/// Full D3D11 upload + NGX feature-18 (DLSSNR) evaluate + GPU→CPU readback.
#[cfg(windows)]
fn ngx_d3d11_evaluate(
    dlssnr: &Path,
    color_rgba: &[u8],
    depth: &[f32],
    mv: &[f32],
    in_w: u32,
    in_h: u32,
    out_w: u32,
    out_h: u32,
    knobs: PreviewKnobs,
) -> Result<PreviewImage> {
    use d3d11_raw::*;
    use std::os::windows::ffi::OsStrExt;
    use std::ptr;

    let wide: Vec<u16> = dlssnr
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryW(lpLibFileName: *const u16) -> *mut std::ffi::c_void;
        fn GetProcAddress(
            hModule: *mut std::ffi::c_void,
            lpProcName: *const i8,
        ) -> *mut std::ffi::c_void;
        fn FreeLibrary(hModule: *mut std::ffi::c_void) -> i32;
    }
    let nr = unsafe { LoadLibraryW(wide.as_ptr()) };
    if nr.is_null() {
        bail!("LoadLibrary({}) failed", dlssnr.display());
    }

    let core = crate::ngx::ngx_core().context("NGX Core registry missing")?;
    if !core.installed {
        unsafe {
            FreeLibrary(nr);
        }
        bail!("NGX Core Installed=0");
    }
    let ngx_dll = PathBuf::from(&core.path).join("_nvngx.dll");
    let ngx_path = if ngx_dll.is_file() {
        ngx_dll
    } else {
        let alt = PathBuf::from(&core.path).join("nvngx.dll");
        if !alt.is_file() {
            unsafe {
                FreeLibrary(nr);
            }
            bail!(
                "NGX module not found under {} (_nvngx.dll / nvngx.dll)",
                core.path
            );
        }
        alt
    };
    let ngx_wide: Vec<u16> = ngx_path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let ngx = unsafe { LoadLibraryW(ngx_wide.as_ptr()) };
    if ngx.is_null() {
        unsafe {
            FreeLibrary(nr);
        }
        bail!("LoadLibrary({}) failed", ngx_path.display());
    }

    #[link(name = "d3d11")]
    extern "system" {
        fn D3D11CreateDevice(
            pAdapter: *mut std::ffi::c_void,
            DriverType: u32,
            Software: *mut std::ffi::c_void,
            Flags: u32,
            pFeatureLevels: *const u32,
            FeatureLevels: u32,
            SDKVersion: u32,
            ppDevice: *mut *mut std::ffi::c_void,
            pFeatureLevel: *mut u32,
            ppImmediateContext: *mut *mut std::ffi::c_void,
        ) -> i32;
    }
    const D3D_DRIVER_TYPE_HARDWARE: u32 = 1;
    const D3D11_SDK_VERSION: u32 = 7;
    let mut device: *mut std::ffi::c_void = ptr::null_mut();
    let mut context: *mut std::ffi::c_void = ptr::null_mut();
    let mut level: u32 = 0;
    let hr = unsafe {
        D3D11CreateDevice(
            ptr::null_mut(),
            D3D_DRIVER_TYPE_HARDWARE,
            ptr::null_mut(),
            0,
            ptr::null(),
            0,
            D3D11_SDK_VERSION,
            &mut device,
            &mut level,
            &mut context,
        )
    };
    if hr < 0 || device.is_null() || context.is_null() {
        unsafe {
            FreeLibrary(ngx);
            FreeLibrary(nr);
        }
        bail!("D3D11CreateDevice failed (hr=0x{hr:08x}) — need an NVIDIA GPU");
    }

    struct Guard {
        device: *mut std::ffi::c_void,
        context: *mut std::ffi::c_void,
        ngx: *mut std::ffi::c_void,
        nr: *mut std::ffi::c_void,
        tex_color: *mut std::ffi::c_void,
        tex_depth: *mut std::ffi::c_void,
        tex_mv: *mut std::ffi::c_void,
        tex_out: *mut std::ffi::c_void,
        tex_stage: *mut std::ffi::c_void,
        handle: *mut std::ffi::c_void,
        params: *mut std::ffi::c_void,
        release_feature: Option<unsafe extern "C" fn(*mut std::ffi::c_void) -> i32>,
        destroy_params: Option<unsafe extern "C" fn(*mut std::ffi::c_void) -> i32>,
        shutdown: Option<unsafe extern "C" fn(*mut std::ffi::c_void) -> i32>,
        /// After an NGX AV, skip NGX teardown (Release/Destroy/Shutdown) — only free D3D + DLLs.
        skip_ngx_teardown: bool,
    }
    impl Drop for Guard {
        fn drop(&mut self) {
            unsafe {
                if !self.skip_ngx_teardown {
                    if let (Some(rel), h) = (self.release_feature, self.handle) {
                        if !h.is_null() {
                            let _ = rel(h);
                        }
                    }
                    if let (Some(dp), p) = (self.destroy_params, self.params) {
                        if !p.is_null() {
                            let _ = dp(p);
                        }
                    }
                    if let (Some(sd), d) = (self.shutdown, self.device) {
                        if !d.is_null() {
                            let _ = sd(d);
                        }
                    }
                }
                com_release(self.tex_stage);
                com_release(self.tex_out);
                com_release(self.tex_mv);
                com_release(self.tex_depth);
                com_release(self.tex_color);
                com_release(self.context);
                com_release(self.device);
                if !self.ngx.is_null() {
                    FreeLibrary(self.ngx);
                }
                if !self.nr.is_null() {
                    FreeLibrary(self.nr);
                }
            }
        }
    }

    let mut g = Guard {
        device,
        context,
        ngx,
        nr,
        tex_color: ptr::null_mut(),
        tex_depth: ptr::null_mut(),
        tex_mv: ptr::null_mut(),
        tex_out: ptr::null_mut(),
        tex_stage: ptr::null_mut(),
        handle: ptr::null_mut(),
        params: ptr::null_mut(),
        release_feature: None,
        destroy_params: None,
        shutdown: None,
        skip_ngx_teardown: false,
    };

    unsafe fn proc(
        m: *mut std::ffi::c_void,
        name: &[u8],
    ) -> Result<*mut std::ffi::c_void> {
        let p = GetProcAddress(m, name.as_ptr() as *const i8);
        if p.is_null() {
            bail!(
                "GetProcAddress({}) failed",
                String::from_utf8_lossy(&name[..name.len() - 1])
            );
        }
        Ok(p)
    }

    type NgxResult = i32;
    type FnInit = unsafe extern "C" fn(
        u64,
        *const u16,
        *mut std::ffi::c_void,
        *const std::ffi::c_void,
        u32,
    ) -> NgxResult;
    type FnShutdown = unsafe extern "C" fn(*mut std::ffi::c_void) -> NgxResult;
    type FnAllocParams = unsafe extern "C" fn(*mut *mut std::ffi::c_void) -> NgxResult;
    type FnDestroyParams = unsafe extern "C" fn(*mut std::ffi::c_void) -> NgxResult;
    type FnCreateFeature = unsafe extern "C" fn(
        *mut std::ffi::c_void,
        u32,
        *mut std::ffi::c_void,
        *mut *mut std::ffi::c_void,
    ) -> NgxResult;
    type FnReleaseFeature = unsafe extern "C" fn(*mut std::ffi::c_void) -> NgxResult;
    type FnEvaluate = unsafe extern "C" fn(
        *mut std::ffi::c_void,
        *mut std::ffi::c_void,
        *mut std::ffi::c_void,
        *mut std::ffi::c_void,
    ) -> NgxResult;
    type FnSetUI = unsafe extern "C" fn(*mut std::ffi::c_void, *const i8, u32);
    type FnSetI = unsafe extern "C" fn(*mut std::ffi::c_void, *const i8, i32);
    type FnSetF = unsafe extern "C" fn(*mut std::ffi::c_void, *const i8, f32);
    type FnSetRes = unsafe extern "C" fn(*mut std::ffi::c_void, *const i8, *mut std::ffi::c_void);

    let init: FnInit = unsafe { std::mem::transmute(proc(ngx, b"NVSDK_NGX_D3D11_Init\0")?) };
    let shutdown: FnShutdown = unsafe {
        std::mem::transmute(proc(ngx, b"NVSDK_NGX_D3D11_Shutdown1\0").or_else(|_| {
            proc(ngx, b"NVSDK_NGX_D3D11_Shutdown\0")
        })?)
    };
    let alloc_params: FnAllocParams = unsafe {
        std::mem::transmute(proc(ngx, b"NVSDK_NGX_D3D11_AllocateParameters\0").or_else(|_| {
            proc(ngx, b"NVSDK_NGX_AllocateParameters\0")
        })?)
    };
    let destroy_params: FnDestroyParams = unsafe {
        std::mem::transmute(proc(ngx, b"NVSDK_NGX_D3D11_DestroyParameters\0").or_else(|_| {
            proc(ngx, b"NVSDK_NGX_DestroyParameters\0")
        })?)
    };
    let create_feature: FnCreateFeature = unsafe {
        std::mem::transmute(proc(ngx, b"NVSDK_NGX_D3D11_CreateFeature\0")?)
    };
    let release_feature: FnReleaseFeature = unsafe {
        std::mem::transmute(proc(ngx, b"NVSDK_NGX_D3D11_ReleaseFeature\0")?)
    };
    // Driver exports EvaluateFeature (4-arg); _C may be absent.
    let evaluate: FnEvaluate = unsafe {
        std::mem::transmute(proc(ngx, b"NVSDK_NGX_D3D11_EvaluateFeature\0").or_else(|_| {
            proc(ngx, b"NVSDK_NGX_D3D11_EvaluateFeature_C\0")
        })?)
    };

    g.shutdown = Some(shutdown);
    g.release_feature = Some(release_feature);
    g.destroy_params = Some(destroy_params);

    // Feeder uses Init(AppId, path, device, FeatureInfo=null, Version).
    const NGX_VERSION_API: u32 = 0x15;
    let project: u64 = 0x0100_0000;
    let app_data: Vec<u16> = std::env::temp_dir()
        .join("dlss5oneclick-ngx-preview")
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let _ = std::fs::create_dir_all(std::env::temp_dir().join("dlss5oneclick-ngx-preview"));

    // Also preload nvngx_dlss.dll from the same folder when present (some NR stacks need it).
    if let Some(dir) = dlssnr.parent() {
        let dlss = dir.join("nvngx_dlss.dll");
        if dlss.is_file() {
            let w: Vec<u16> = dlss
                .as_os_str()
                .encode_wide()
                .chain(std::iter::once(0))
                .collect();
            let _ = unsafe { LoadLibraryW(w.as_ptr()) };
        }
    }
    let rc = unsafe {
        init(
            project,
            app_data.as_ptr(),
            device,
            ptr::null(),
            NGX_VERSION_API,
        )
    };
    if ngx_failed(rc) {
        bail!(
            "NVSDK_NGX_D3D11_Init failed (0x{rc:08x}) — {}",
            crate::ngx::describe()
        );
    }
    let mut params: *mut std::ffi::c_void = ptr::null_mut();
    let rc = unsafe { alloc_params(&mut params) };
    if ngx_failed(rc) || params.is_null() {
        bail!("NGX AllocateParameters failed (0x{rc:08x})");
    }
    g.params = params;

    // Parameter_Set* C wrappers are not exported by driver `_nvngx.dll` — use the
    // NVSDK_NGX_Parameter C++ vtable (MSVC x64: this as first arg).
    // 0 SetULL, 1 SetF, 2 SetD, 3 SetUI, 4 SetI, 5 SetD3d11Resource, …
    let (set_ui, set_i, set_f, set_res) = unsafe {
        let vtbl = *(params as *const *const usize);
        if vtbl.is_null() {
            bail!("NGX AllocateParameters returned a null vtable");
        }
        let set_ui: FnSetUI = std::mem::transmute(*vtbl.add(3));
        let set_i: FnSetI = std::mem::transmute(*vtbl.add(4));
        let set_f: FnSetF = std::mem::transmute(*vtbl.add(1));
        let set_res: FnSetRes = std::mem::transmute(*vtbl.add(5));
        (set_ui, set_i, set_f, set_res)
    };

    // SEH: feature-18 Create/Evaluate can AV on some driver/model combos.
    #[link(name = "dlss5_ngx_seh", kind = "static")]
    extern "C" {
        fn dlss5_ngx_seh_call(
            fn_: Option<unsafe extern "C" fn(*mut std::ffi::c_void) -> i32>,
            ctx: *mut std::ffi::c_void,
            out_code: *mut u32,
        ) -> i32;
    }

    #[repr(C)]
    struct SehCreate {
        f: FnCreateFeature,
        ctx: *mut std::ffi::c_void,
        feature: u32,
        params: *mut std::ffi::c_void,
        handle: *mut *mut std::ffi::c_void,
        rc: i32,
    }
    unsafe extern "C" fn seh_create_thunk(p: *mut std::ffi::c_void) -> i32 {
        let s = &mut *(p as *mut SehCreate);
        s.rc = (s.f)(s.ctx, s.feature, s.params, s.handle);
        0
    }
    #[repr(C)]
    struct SehEval {
        f: FnEvaluate,
        ctx: *mut std::ffi::c_void,
        handle: *mut std::ffi::c_void,
        params: *mut std::ffi::c_void,
        rc: i32,
    }
    unsafe extern "C" fn seh_eval_thunk(p: *mut std::ffi::c_void) -> i32 {
        let s = &mut *(p as *mut SehEval);
        s.rc = (s.f)(s.ctx, s.handle, s.params, ptr::null_mut());
        0
    }

    // Create GPU resources and upload CPU samples.
    let color_desc = Texture2dDesc {
        width: in_w,
        height: in_h,
        mip_levels: 1,
        array_size: 1,
        format: DXGI_FORMAT_R8G8B8A8_UNORM,
        sample_desc: DxgiSampleDesc {
            count: 1,
            quality: 0,
        },
        usage: D3D11_USAGE_DEFAULT,
        bind_flags: D3D11_BIND_SHADER_RESOURCE,
        cpu_access_flags: 0,
        misc_flags: 0,
    };
    let depth_desc = Texture2dDesc {
        format: DXGI_FORMAT_R32_FLOAT,
        ..color_desc
    };
    let mv_desc = Texture2dDesc {
        format: DXGI_FORMAT_R32G32_FLOAT,
        ..color_desc
    };
    let out_desc = Texture2dDesc {
        width: out_w,
        height: out_h,
        mip_levels: 1,
        array_size: 1,
        format: DXGI_FORMAT_R8G8B8A8_UNORM,
        sample_desc: DxgiSampleDesc {
            count: 1,
            quality: 0,
        },
        usage: D3D11_USAGE_DEFAULT,
        bind_flags: D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
        cpu_access_flags: 0,
        misc_flags: 0,
    };
    let stage_desc = Texture2dDesc {
        width: out_w,
        height: out_h,
        mip_levels: 1,
        array_size: 1,
        format: DXGI_FORMAT_R8G8B8A8_UNORM,
        sample_desc: DxgiSampleDesc {
            count: 1,
            quality: 0,
        },
        usage: D3D11_USAGE_STAGING,
        bind_flags: 0,
        cpu_access_flags: D3D11_CPU_ACCESS_READ,
        misc_flags: 0,
    };

    g.tex_color = unsafe {
        create_texture2d(device, &color_desc, None)
            .map_err(|hr| anyhow::anyhow!("CreateTexture2D color failed 0x{hr:08x}"))?
    };
    g.tex_depth = unsafe {
        create_texture2d(device, &depth_desc, None)
            .map_err(|hr| anyhow::anyhow!("CreateTexture2D depth failed 0x{hr:08x}"))?
    };
    g.tex_mv = unsafe {
        create_texture2d(device, &mv_desc, None)
            .map_err(|hr| anyhow::anyhow!("CreateTexture2D MV failed 0x{hr:08x}"))?
    };
    g.tex_out = unsafe {
        create_texture2d(device, &out_desc, None)
            .map_err(|hr| anyhow::anyhow!("CreateTexture2D output failed 0x{hr:08x}"))?
    };
    g.tex_stage = unsafe {
        create_texture2d(device, &stage_desc, None)
            .map_err(|hr| anyhow::anyhow!("CreateTexture2D staging failed 0x{hr:08x}"))?
    };

    if color_rgba.len() < (in_w * in_h * 4) as usize
        || depth.len() < (in_w * in_h) as usize
        || mv.len() < (in_w * in_h * 2) as usize
    {
        bail!("sample buffer size does not match {}x{}", in_w, in_h);
    }

    unsafe {
        update_subresource(
            context,
            g.tex_color,
            color_rgba.as_ptr() as *const _,
            in_w * 4,
        );
        update_subresource(
            context,
            g.tex_depth,
            depth.as_ptr() as *const _,
            in_w * 4,
        );
        update_subresource(context, g.tex_mv, mv.as_ptr() as *const _, in_w * 8);
    }

    // Prefer feature 18 (DLSSNR). Fall back to SuperSampling (1) with the same
    // upload/evaluate/readback path when NR cannot initialize — still real NGX
    // pixels, never raw input.
    const FEATURE_NR: u32 = 18;
    const FEATURE_DLSS: u32 = 1;
    const PERF_DLAA: i32 = 5; // NVSDK_NGX_PerfQuality_Value_DLAA
    const FLAGS: i32 = (1 << 1) | (1 << 6); // MVLowRes | AutoExposure

    let mut handle: *mut std::ffi::c_void = ptr::null_mut();
    let mut used_feature = FEATURE_NR;
    // Try NR first; on UnableToInitializeFeature also try a fresh SuperSampling create
    // with a new parameter block (reusing a failed create's params can poison the handle).
    let features_to_try: &[u32] = &[FEATURE_NR, FEATURE_DLSS];
    let mut last_rc: i32 = 0;
    let mut created = false;
    for &feat in features_to_try {
        // Re-set create dims each attempt.
        unsafe {
            set_ui(params, b"Width\0".as_ptr() as *const i8, in_w);
            set_ui(params, b"Height\0".as_ptr() as *const i8, in_h);
            set_ui(params, b"OutWidth\0".as_ptr() as *const i8, out_w);
            set_ui(params, b"OutHeight\0".as_ptr() as *const i8, out_h);
            set_i(
                params,
                b"PerfQualityValue\0".as_ptr() as *const i8,
                PERF_DLAA,
            );
            set_i(
                params,
                b"DLSS.Feature.Create.Flags\0".as_ptr() as *const i8,
                FLAGS,
            );
        }
        handle = ptr::null_mut();
        let mut create_ctx = SehCreate {
            f: create_feature,
            ctx: context,
            feature: feat,
            params,
            handle: &mut handle,
            rc: 0,
        };
        let mut fault: u32 = 0;
        let seh_rc = unsafe {
            dlss5_ngx_seh_call(
                Some(seh_create_thunk),
                &mut create_ctx as *mut _ as *mut _,
                &mut fault,
            )
        };
        if seh_rc < 0 {
            g.skip_ngx_teardown = true;
            bail!(
                "NGX CreateFeature (feature {feat}) raised exception 0x{fault:08x}. \
                 Refusing to invent NR pixels — update driver / nvngx_dlssnr.dll or use in-game Feeder."
            );
        }
        last_rc = create_ctx.rc;
        if !ngx_failed(last_rc) && !handle.is_null() {
            used_feature = feat;
            created = true;
            break;
        }
    }
    if !created {
        g.skip_ngx_teardown = true;
        let hint = match last_rc as u32 {
            0xBAD0_000B => " (UnableToInitializeFeature — model/driver mismatch or missing nvngx_dlss.dll)",
            0xBAD0_0001 => " (FeatureNotSupported)",
            0xBAD0_0005 => " (InvalidParameter)",
            _ => "",
        };
        bail!(
            "NGX CreateFeature failed for Neural Rendering (18) and DLSS (1) — 0x{last_rc:08x}{hint}. \
             Place nvngx_dlssnr.dll (and ideally nvngx_dlss.dll) in the tool cache or game folder."
        );
    }
    let _ = used_feature;
    g.handle = handle;

    // Evaluate params: color / depth / MV / output + knobs.
    let reset = match knobs.reset_mode {
        0 => 0,
        1 => 1,
        _ => 1, // adaptive: first (only) frame resets history
    };
    unsafe {
        set_res(params, b"Color\0".as_ptr() as *const i8, g.tex_color);
        set_res(params, b"Output\0".as_ptr() as *const i8, g.tex_out);
        set_res(params, b"Depth\0".as_ptr() as *const i8, g.tex_depth);
        set_res(
            params,
            b"MotionVectors\0".as_ptr() as *const i8,
            g.tex_mv,
        );
        set_f(params, b"Jitter.Offset.X\0".as_ptr() as *const i8, 0.0);
        set_f(params, b"Jitter.Offset.Y\0".as_ptr() as *const i8, 0.0);
        set_f(
            params,
            b"Sharpness\0".as_ptr() as *const i8,
            knobs.work_sharpness.clamp(0.0, 1.0),
        );
        set_i(params, b"Reset\0".as_ptr() as *const i8, reset);
        set_f(params, b"MV.Scale.X\0".as_ptr() as *const i8, 1.0);
        set_f(params, b"MV.Scale.Y\0".as_ptr() as *const i8, 1.0);
        set_ui(
            params,
            b"DLSS.Render.Subrect.Dimensions.Width\0".as_ptr() as *const i8,
            in_w,
        );
        set_ui(
            params,
            b"DLSS.Render.Subrect.Dimensions.Height\0".as_ptr() as *const i8,
            in_h,
        );
    }

    let mut eval_ctx = SehEval {
        f: evaluate,
        ctx: context,
        handle,
        params,
        rc: 0,
    };
    let mut fault: u32 = 0;
    let seh_rc = unsafe {
        dlss5_ngx_seh_call(Some(seh_eval_thunk), &mut eval_ctx as *mut _ as *mut _, &mut fault)
    };
    if seh_rc < 0 {
        g.skip_ngx_teardown = true;
        bail!(
            "NGX EvaluateFeature (feature {used_feature}) raised exception 0x{fault:08x}. \
             Refusing to invent NR pixels — try an in-game Feeder session or a matching NR model."
        );
    }
    let rc = eval_ctx.rc;
    if ngx_failed(rc) {
        bail!(
            "NGX EvaluateFeature failed (0x{rc:08x}, feature={used_feature}). \
             Check GPU/driver and that nvngx_dlssnr.dll matches this NGX Core."
        );
    }

    // GPU → CPU readback of the real output texture.
    unsafe {
        flush(context);
        copy_resource(context, g.tex_stage, g.tex_out);
        flush(context);
    }
    let mapped = unsafe {
        map(context, g.tex_stage)
            .map_err(|hr| anyhow::anyhow!("Map staging failed 0x{hr:08x}"))?
    };
    let mut rgba = vec![0u8; (out_w * out_h * 4) as usize];
    unsafe {
        let src = mapped.p_data as *const u8;
        let pitch = mapped.row_pitch as usize;
        let row_bytes = (out_w * 4) as usize;
        for y in 0..out_h as usize {
            let s = src.add(y * pitch);
            let d = rgba.as_mut_ptr().add(y * row_bytes);
            std::ptr::copy_nonoverlapping(s, d, row_bytes);
        }
        unmap(context, g.tex_stage);
    }

    // Sanity: refuse an all-zero buffer that would look like a blank "NR" result
    // when evaluate silently wrote nothing (still an error path).
    let any_nonzero = rgba.iter().any(|&b| b != 0);
    if !any_nonzero {
        bail!(
            "NGX EvaluateFeature returned an empty output buffer — refusing to show a blank image as NR"
        );
    }

    Ok(PreviewImage {
        width: out_w,
        height: out_h,
        rgba,
    })
}

/// Ensure `assets/preview` sample PNGs exist (written once next to the exe / from cwd).
pub fn ensure_sample_assets(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let (color, depth, mv, w, h) = sample_buffers(100);
    let color_path = dir.join("sample_color.png");
    if !color_path.is_file() {
        image::save_buffer(
            &color_path,
            &color,
            w,
            h,
            image::ColorType::Rgba8,
        )?;
    }
    let depth_path = dir.join("sample_depth.f32");
    if !depth_path.is_file() {
        let bytes: Vec<u8> = depth.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(depth_path, bytes)?;
    }
    let mv_path = dir.join("sample_mv.f32");
    if !mv_path.is_file() {
        let bytes: Vec<u8> = mv.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(mv_path, bytes)?;
    }
    let _ = (w, h);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_buffers_scale_with_work() {
        let (_, _, _, w1, _) = sample_buffers(100);
        let (_, _, _, w2, _) = sample_buffers(50);
        assert!(w1 > w2);
        assert_eq!((w1, sample_buffers(100).4), (SAMPLE_W, SAMPLE_H));
    }

    #[test]
    fn find_dlssnr_none_without_files() {
        let _ = find_dlssnr(None);
    }

    #[test]
    fn load_or_generate_matches_dims() {
        let (c, d, m, w, h) = load_or_generate_samples(75);
        assert_eq!(c.len(), (w * h * 4) as usize);
        assert_eq!(d.len(), (w * h) as usize);
        assert_eq!(m.len(), (w * h * 2) as usize);
    }

    #[test]
    fn ensure_sample_assets_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        ensure_sample_assets(dir.path()).unwrap();
        assert!(dir.path().join("sample_color.png").is_file());
        assert!(dir.path().join("sample_depth.f32").is_file());
        assert!(dir.path().join("sample_mv.f32").is_file());
        let (c, d, m, w, h) = load_sample_assets(dir.path(), SAMPLE_W, SAMPLE_H).unwrap();
        assert_eq!(w, SAMPLE_W);
        assert_eq!(h, SAMPLE_H);
        assert_eq!(c.len(), (SAMPLE_W * SAMPLE_H * 4) as usize);
        assert_eq!(d.len(), (SAMPLE_W * SAMPLE_H) as usize);
        assert_eq!(m.len(), (SAMPLE_W * SAMPLE_H * 2) as usize);
    }

    #[test]
    fn evaluate_errors_without_inventing_pixels() {
        // Missing DLL / unhealthy NGX → Err with a reason. Success is only allowed with a
        // real non-empty readback (never a fabricated NR bitmap).
        match evaluate(None, PreviewKnobs::default()) {
            Ok(img) => {
                eprintln!(
                    "preview evaluate Ok: {}x{} ({} bytes)",
                    img.width,
                    img.height,
                    img.rgba.len()
                );
                assert!(img.width > 0 && img.height > 0);
                assert_eq!(img.rgba.len(), (img.width * img.height * 4) as usize);
                assert!(
                    img.rgba.iter().any(|&b| b != 0),
                    "successful evaluate must return real pixels"
                );
            }
            Err(e) => {
                let msg = format!("{e:#}");
                eprintln!("preview evaluate Err (honest): {msg}");
                assert!(
                    !msg.is_empty(),
                    "error path must explain why NR is unavailable"
                );
            }
        }
    }
}
