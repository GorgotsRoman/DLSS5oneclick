fn main() {
    println!("cargo:rerun-if-changed=assets/icon.ico");
    println!("cargo:rerun-if-changed=assets/patches/DLSS5_Feed.fx");
    println!("cargo:rerun-if-changed=assets/patches/dlss5-feed.addon64");
    println!("cargo:rerun-if-changed=src/native/ngx_seh.c");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        // Hybrid-graphics machines pick the GPU for a process by looking for
        // these two exports in the exe (NVIDIA Optimus and AMD PowerXpress).
        // Without them the window can land on the integrated part, and on one
        // machine that meant the GUI died inside AMD's OpenGL driver before
        // main() ran -- no window, no log, nothing to report (#32, #23).
        println!("cargo:rustc-link-arg-bins=/EXPORT:NvOptimusEnablement,DATA");
        println!("cargo:rustc-link-arg-bins=/EXPORT:AmdPowerXpressRequestHighPerformance,DATA");
        cc::Build::new()
            .file("src/native/ngx_seh.c")
            .compile("dlss5_ngx_seh");
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/icon.ico");
        res.set("ProductName", "DLSS5oneclick");
        res.set(
            "FileDescription",
            "One-click DLSS 5 setup for games without DLSS",
        );
        if let Err(e) = res.compile() {
            println!("cargo:warning=winresource failed: {e}");
        }
    }
}