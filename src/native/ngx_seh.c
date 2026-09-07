#include <windows.h>

typedef int (__cdecl *dlss5_ngx_thunk)(void *ctx);

/* Catch NGX CreateFeature / EvaluateFeature AVs (common on feature 18) and
   surface them as a return code instead of killing the process. */
int dlss5_ngx_seh_call(dlss5_ngx_thunk fn, void *ctx, unsigned long *out_code)
{
    if (out_code) *out_code = 0;
    __try {
        return fn(ctx);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        if (out_code) *out_code = GetExceptionCode();
        return -1;
    }
}
