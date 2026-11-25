//! Reads of title memory at the speed of a plain copy, that still report a
//! page that is not there instead of faulting.
//!
//! Host copies of title memory were read through the kernel
//! (`ReadProcessMemory`), because the title may unmap or free a page at any
//! time and a plain copy of it would crash. That costs a system call per
//! copy and copies at a third of the speed of the processor's own: in the
//! world (run "gpu-relay") the submitting thread spent 36 ms a frame moving
//! some 125 MB that way, 94 ms a frame in the kernel in all.
//!
//! Here the copy is one `rep movsb` in a routine of our own. If a source
//! page is not there, the processor faults at that one instruction; a
//! vectored handler sees the fault at that address, and only there, and
//! resumes the routine at its failure exit, which says so. That is what
//! kernels do for copies from user memory (an exception table).
//!
//! What is left to the caller: the destination must be mapped writable
//! memory of ours (a fault on it is reported the same way, as a failed
//! read, and the caller's slower path then finds out which page it was).

#[cfg(target_arch = "x86_64")]
std::arch::global_asm!(
    ".globl victory_safecopy",
    ".globl victory_safecopy_move",
    ".globl victory_safecopy_fault",
    // (rcx destination, rdx source, r8 length) -> eax 1 copied, 0 faulted.
    "victory_safecopy:",
    "    push rdi",
    "    push rsi",
    "    mov rdi, rcx",
    "    mov rsi, rdx",
    "    mov rcx, r8",
    "victory_safecopy_move:",
    "    rep movsb",
    "    pop rsi",
    "    pop rdi",
    "    mov eax, 1",
    "    ret",
    // Reached only from the handler, with the stack as at the move.
    "victory_safecopy_fault:",
    "    cld",
    "    pop rsi",
    "    pop rdi",
    "    xor eax, eax",
    "    ret",
);

#[cfg(target_arch = "x86_64")]
unsafe extern "C" {
    fn victory_safecopy(dst: *mut u8, src: *const u8, len: usize) -> u32;
    static victory_safecopy_move: u8;
    static victory_safecopy_fault: u8;
}

/// Registers the handler once. Taken ahead of the others: it answers only
/// for a fault at the routine's one move.
#[cfg(target_arch = "x86_64")]
fn install() {
    use windows_sys::Win32::Foundation::{
        EXCEPTION_ACCESS_VIOLATION, EXCEPTION_IN_PAGE_ERROR, NTSTATUS, STATUS_GUARD_PAGE_VIOLATION,
    };
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_fault(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched and the
        // faulting thread's context.
        let (record, context) = unsafe { (&*(*info).ExceptionRecord, &mut *(*info).ContextRecord) };
        let code = record.ExceptionCode;
        let ours = [EXCEPTION_ACCESS_VIOLATION, EXCEPTION_IN_PAGE_ERROR, STATUS_GUARD_PAGE_VIOLATION]
            .iter()
            .any(|&c| c as NTSTATUS == code);
        // The addresses of two labels; nothing is read.
        let (moving, fault) = (&raw const victory_safecopy_move, &raw const victory_safecopy_fault);
        if !ours || !std::ptr::addr_eq(record.ExceptionAddress, moving) {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        // The caller's slower path takes over and is counted there
        // (`copystat`'s fallbacks).
        context.Rip = fault as u64;
        EXCEPTION_CONTINUE_EXECUTION
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only moves a thread of ours out of the routine.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_fault));
    });
}

/// Copies `len` bytes from `src` to `dst`; false if a page of either could
/// not be reached (then part of `dst` may have been written).
///
/// # Safety
/// `dst` must be `len` bytes of writable memory of ours that nothing else
/// uses during the copy; `src` may be anything.
pub(crate) unsafe fn copy(dst: *mut u8, src: *const u8, len: usize) -> bool {
    if len == 0 {
        return true;
    }
    #[cfg(target_arch = "x86_64")]
    {
        install();
        // SAFETY: the caller vouches for the destination; a fault on the
        // source ends the routine through the handler.
        unsafe { victory_safecopy(dst, src, len) != 0 }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (dst, src);
        false
    }
}

#[cfg(test)]
mod tests {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        VirtualProtect,
    };

    #[test]
    fn a_readable_source_is_copied_whole() {
        let src: Vec<u8> = (0..100_000u32).map(|i| (i * 7) as u8).collect();
        let mut dst = vec![0u8; src.len()];
        // SAFETY: a destination of ours of that length.
        assert!(unsafe { super::copy(dst.as_mut_ptr(), src.as_ptr(), src.len()) });
        assert_eq!(dst, src);
    }

    #[test]
    fn a_page_that_is_not_there_is_reported_not_faulted() {
        const PAGE: usize = 4096;
        // SAFETY: a fresh allocation of ours, freed below.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), 4 * PAGE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE).cast::<u8>();
            assert!(!base.is_null());
            base.write_bytes(5, 4 * PAGE);
            let mut old = 0;
            assert_ne!(VirtualProtect(base.add(2 * PAGE).cast(), PAGE, PAGE_NOACCESS, &mut old), 0);
            let mut dst = vec![0u8; 4 * PAGE];
            assert!(!super::copy(dst.as_mut_ptr(), base, 4 * PAGE));
            // What came before the hole was copied; the thread goes on.
            assert!(dst[..2 * PAGE].iter().all(|&b| b == 5));
            assert!(super::copy(dst.as_mut_ptr(), base, 2 * PAGE));
            assert!(super::copy(dst.as_mut_ptr(), base.add(3 * PAGE), PAGE));
            VirtualFree(base.cast(), 0, MEM_RELEASE);
        }
    }

    #[test]
    fn copies_on_many_threads_fault_alone() {
        const PAGE: usize = 4096;
        // SAFETY: a fresh allocation of ours, freed after the threads end.
        let base = unsafe {
            let base = VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE).cast::<u8>();
            let mut old = 0;
            VirtualProtect(base.add(PAGE).cast(), PAGE, PAGE_NOACCESS, &mut old);
            base as usize
        };
        let threads: Vec<_> = (0..8)
            .map(|i| {
                std::thread::spawn(move || {
                    let mut dst = vec![0u8; 2 * PAGE];
                    for _ in 0..200 {
                        let whole = i % 2 == 0;
                        let len = if whole { PAGE } else { 2 * PAGE };
                        // SAFETY: our own destination.
                        let ok = unsafe { super::copy(dst.as_mut_ptr(), base as *const u8, len) };
                        assert_eq!(ok, whole);
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        // SAFETY: ours, no thread uses it any more.
        unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
    }
}
