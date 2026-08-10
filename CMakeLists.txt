cmake_minimum_required(VERSION 3.26)

project(ct_workstation LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC OFF)
set(CMAKE_AUTORCC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

find_package(Qt6 6.6 REQUIRED COMPONENTS Core Gui Widgets Network)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(PVCAM_SDK_ROOT "C:/Program Files/Photometrics/PVCamSDK" CACHE PATH "Teledyne Photometrics PVCAM SDK root")
set(PVCAM_INCLUDE_DIR "${PVCAM_SDK_ROOT}/Inc")
set(PVCAM_LIBRARY "${PVCAM_SDK_ROOT}/Lib/amd64/pvcam64.lib")
set(PVCAM_FOUND OFF)
if (EXISTS "${PVCAM_INCLUDE_DIR}/pvcam.h" AND EXISTS "${PVCAM_INCLUDE_DIR}/master.h" AND EXISTS "${PVCAM_LIBRARY}")
    set(PVCAM_FOUND ON)
endif()

qt_standard_project_setup()

add_library(ct_workstation_cuda STATIC
    src/core/recon/Bp3dCudaBackend.h
    src/core/recon/Bp3dCudaBackend.cpp
    src/core/recon/GpuProjectionBatch.h
    src/core/recon/GpuProjectionBatch.cpp
    src/core/recon/GpuVectorBuffer.h
    src/core/recon/GpuVectorBuffer.cpp
    src/core/recon/IReconstructionVolume.h
    src/core/recon/GpuVolumeBuffer.h
    src/core/recon/GpuVolumeBuffer.cpp
    src/cuda/bp3d_kernel.cu
    src/cuda/preprocess_kernels.h
    src/cuda/preprocess_kernels.cu
    src/cuda/offline_tomo_kernels.h
    src/cuda/offline_tomo_kernels.cu
    src/cuda/stripe_removal_kernels.h
    src/cuda/stripe_removal_kernels.cu
)

target_include_directories(ct_workstation_cuda
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ct_workstation_cuda
    PUBLIC
        Qt6::Core
        ${OpenCV_LIBS}
        CUDA::cudart
        CUDA::cufft
)

set_target_properties(ct_workstation_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(ct_workstation_cuda
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

add_library(ct_workstation_core STATIC
    src/app/ReconstructionController.h
    src/app/ReconstructionController.cpp
    src/core/acq/CtAcquisitionTypes.h
    src/core/acq/ContinuousTiming.h
    src/core/acq/ContinuousTiming.cpp
    src/core/acq/FrameBufferPool.h
    src/core/acq/FrameBufferPool.cpp
    src/core/acq/CtReplayAcquisitionWorker.h
    src/core/acq/CtReplayAcquisitionWorker.cpp
    src/core/acq/CtTiffWriter.h
    src/core/acq/CtTiffWriter.cpp
    src/core/config/AppConfig.h
    src/core/config/AppConfig.cpp
    src/core/common/GeometryVectorGenerator.h
    src/core/common/GeometryVectorGenerator.cpp
    src/core/common/PreviewRenderWorker.h
    src/core/common/PreviewRenderWorker.cpp
    src/core/common/PreviewSliceRenderer.h
    src/core/common/PreviewSliceRenderer.cpp
    src/core/common/ProjectionTypes.h
    src/core/cor/CorEstimatorFft.h
    src/core/cor/CorEstimatorFft.cpp
    src/core/io/FileProjectionSource.h
    src/core/io/FileProjectionSource.cpp
    src/core/io/ImageFileIo.h
    src/core/io/ImageFileIo.cpp
    src/hardware/camera/BoundedFrameQueue.h
    src/hardware/camera/BoundedFrameQueue.cpp
    src/hardware/camera/CameraFrame.h
    src/hardware/camera/PvcamCameraWorker.h
    src/hardware/camera/PvcamCameraWorker.cpp
    src/hardware/kohzu/KohzuController.h
    src/hardware/kohzu/KohzuController.cpp
    src/core/preprocess/FbpFilter1D.h
    src/core/preprocess/FbpFilter1D.cpp
    src/core/preprocess/CudaFftPlanCache.h
    src/core/preprocess/CudaFftPlanCache.cpp
    src/core/preprocess/ImageNormalizer.h
    src/core/preprocess/ImageNormalizer.cpp
    src/core/preprocess/PaganinProcessor.h
    src/core/preprocess/PaganinProcessor.cpp
    src/core/recon/CpuVolumeBuffer.h
    src/core/recon/CpuVolumeBuffer.cpp
    src/core/recon/ReconstructionRuntimeCache.h
    src/core/recon/ReconstructionRuntimeCache.cpp
    src/core/recon/OfflineTomoReconstruction.h
    src/core/recon/OfflineTomoReconstruction.cpp
    src/core/recon/ReconstructionWorker.h
    src/core/recon/ReconstructionWorker.cpp
    src/core/save/SaveWorker.h
    src/core/save/SaveWorker.cpp
)

target_include_directories(ct_workstation_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ct_workstation_core
    PUBLIC
        Qt6::Core
        Qt6::Gui
        Qt6::Network
        ${OpenCV_LIBS}
        ct_workstation_cuda
)

if (PVCAM_FOUND)
    target_compile_definitions(ct_workstation_core
        PUBLIC
            CT_HAS_PVCAM
    )
    target_include_directories(ct_workstation_core
        PUBLIC
            "${PVCAM_INCLUDE_DIR}"
    )
    target_link_libraries(ct_workstation_core
        PUBLIC
            "${PVCAM_LIBRARY}"
    )
endif()

qt_add_executable(ct_workstation_app
    src/app/main.cpp
    src/ui/ContrastHistogramWidget.h
    src/ui/ContrastHistogramWidget.cpp
    src/ui/MainWindow.h
    src/ui/MainWindow.cpp
    src/ui/ZoomPanImageView.h
    src/ui/ZoomPanImageView.cpp
)

target_link_libraries(ct_workstation_app
    PRIVATE
        Qt6::Core
        Qt6::Gui
        Qt6::Widgets
        ct_workstation_core
)

if (WIN32)
    get_target_property(QT_QMAKE_EXECUTABLE Qt6::qmake IMPORTED_LOCATION)
    if (QT_QMAKE_EXECUTABLE)
        get_filename_component(QT_BIN_DIR "${QT_QMAKE_EXECUTABLE}" DIRECTORY)
        set(WINDEPLOYQT_EXECUTABLE "${QT_BIN_DIR}/windeployqt.exe")
        set_target_properties(ct_workstation_app PROPERTIES
            VS_DEBUGGER_ENVIRONMENT "PATH=${QT_BIN_DIR};%PATH%"
        )

        if (EXISTS "${WINDEPLOYQT_EXECUTABLE}")
            add_custom_command(TARGET ct_workstation_app POST_BUILD
                COMMAND "${WINDEPLOYQT_EXECUTABLE}"
                    --dir "$<TARGET_FILE_DIR:ct_workstation_app>"
                    --no-compiler-runtime
                    --no-translations
                    --no-system-d3d-compiler
                    "$<TARGET_FILE:ct_workstation_app>"
                COMMENT "Deploying Qt runtime with windeployqt"
                VERBATIM
            )
        endif()
    endif()
endif()

enable_testing()

add_executable(ct_workstation_tests
    tests/core_tests.cpp
)

target_include_directories(ct_workstation_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ct_workstation_tests
    PRIVATE
        Qt6::Core
        ${OpenCV_LIBS}
        ct_workstation_core
)

add_test(NAME ct_workstation_tests COMMAND ct_workstation_tests)

add_executable(ct_reconstruct_headless
    tools/ct_reconstruct_headless.cpp
)

target_include_directories(ct_reconstruct_headless
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ct_reconstruct_headless
    PRIVATE
        Qt6::Core
        ${OpenCV_LIBS}
        ct_workstation_core
)

add_executable(ct_pipeline_replay
    tools/ct_pipeline_replay.cpp
)

target_include_directories(ct_pipeline_replay
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ct_pipeline_replay
    PRIVATE
        Qt6::Core
        ${OpenCV_LIBS}
        ct_workstation_core
)

if (PVCAM_FOUND)
    add_executable(pvcam_smoke_test
        tools/pvcam_smoke_test.cpp
    )

    target_include_directories(pvcam_smoke_test
        PRIVATE
            "${PVCAM_INCLUDE_DIR}"
    )

    target_link_libraries(pvcam_smoke_test
        PRIVATE
            "${PVCAM_LIBRARY}"
    )

    add_executable(pvcam_capture_test
        tools/pvcam_capture_test.cpp
    )

    target_include_directories(pvcam_capture_test
        PRIVATE
            "${PVCAM_INCLUDE_DIR}"
            ${OpenCV_INCLUDE_DIRS}
    )

    target_link_libraries(pvcam_capture_test
        PRIVATE
            "${PVCAM_LIBRARY}"
            ${OpenCV_LIBS}
    )
else()
    message(WARNING "PVCAM SDK not found. Set PVCAM_SDK_ROOT to build pvcam_smoke_test.")
endif()
