#pragma once

// Process-wide reuse of large cv::Mat buffers.
//
// Every frame allocates the same large buffers again (grey and float frames,
// gradients, blurs, eigenvalue maps, recovery variants). Most of them (1080p
// frames and float images, megabytes each) come straight from the OS and go
// back to it on release, so every frame pays a demand-zero page fault per page
// on first touch. The caching allocator keeps released large blocks and hands
// them out again (size threshold, bound and size classes: see the constants
// in mat_allocator.cpp).
//
// installCachingMatAllocator() makes it the default Mat allocator:
//  - Results are unchanged: OpenCV never relies on the content of a new Mat
//    (fastMalloc does not clear it either).
//  - Invariant for all code in this module: a new Mat of 256 KB or more holds
//    stale data from earlier frames, not the zeroed pages a fresh block from
//    the OS happens to have. Write every element before reading it; use
//    cv::Mat::zeros (or setTo) where a zero background is needed.
//  - Thread-safe and process-wide; it belongs to no engine, so destroying an
//    engine leaves the cached blocks to the next one.
//  - Never destroyed: every Mat returns its block to the allocator that
//    created it, and Mats can outlive any static destructor.
//  - Only the OpenCV this module links is affected; Python's cv2 ships its
//    own.
//  - Call it before any other thread uses OpenCV (cv::Mat::setDefaultAllocator
//    is an unsynchronised global); the Python module does so at import.
//    Idempotent.

namespace hydramarker {

void installCachingMatAllocator();

}  // namespace hydramarker
