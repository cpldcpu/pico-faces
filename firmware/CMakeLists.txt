cmake_minimum_required(VERSION 3.13)

set(PICO_BOARD pico2 CACHE STRING "")
set(PICO_PLATFORM rp2350-arm-s CACHE STRING "")

include($ENV{PICO_SDK_PATH}/external/pico_sdk_import.cmake)
include($ENV{PICO_EXTRAS_PATH}/external/pico_extras_import.cmake)

project(pico_faces C CXX ASM)
pico_sdk_init()

# RF_VGA=2 (default): interrupt-free scanout (vga_noirq.c) - static DMA
# rings feeding scanvideo's PIO programs; zero display IRQs/CPU.
# RF_VGA=1: the original scanvideo path (IRQ-driven, kept as fallback).
if(NOT DEFINED RF_VGA)
    set(RF_VGA 2)
endif()

# model blob baked into flash (.rodata); override with -DMODEL_BIN=...
if(NOT MODEL_BIN)
    set(MODEL_BIN ${CMAKE_CURRENT_SOURCE_DIR}/../build/model_random.bin)
endif()

# RF_STAGE_DMA=1: overlap weight staging with compute via paced DMA
# (firmware/stage.c). Default 0 = synchronous memcpy staging (the weak
# implementation in kernels_ref.c), byte-identical results either way.
if(NOT DEFINED RF_STAGE_DMA)
    set(RF_STAGE_DMA 0)
endif()

add_executable(pico_faces
    main.c
    par.c
    model_blob.S
    ../engine/src/graph.c
    ../engine/src/dit.c
    ../engine/src/vae_dec.c
    ../engine/src/hires.c
    ../engine/src/kernels_ref.c
    ../engine/src/prng.c
)
# per-model rf_cfg.h sits next to the blob; must come before the engine
# default on the include path (rf_model.h uses an angle include)
if(NOT RF_CFG_DIR)
    get_filename_component(RF_CFG_DIR ${MODEL_BIN} DIRECTORY)
endif()
target_include_directories(pico_faces PRIVATE ${RF_CFG_DIR} ../engine/include)
# engine callbacks run ~1.5KB of locals (softmax tables) on both stacks
target_compile_definitions(pico_faces PRIVATE
    PICO_STACK_SIZE=0x1000 PICO_CORE1_STACK_SIZE=0x1000)
set_source_files_properties(model_blob.S PROPERTIES
    COMPILE_DEFINITIONS "MODEL_BIN=${MODEL_BIN}"
    OBJECT_DEPENDS "${MODEL_BIN}")

# RF_SYS_KHZ: 300000 default (vreg 1.30V, QMI div 4 keeps QSPI at 75 MHz);
# build with -DRF_SYS_KHZ=150000 if a board proves unstable
if(NOT RF_SYS_KHZ)
    set(RF_SYS_KHZ 300000)
endif()
target_compile_definitions(pico_faces PRIVATE
    RF_SYS_KHZ=${RF_SYS_KHZ} RF_HOT_IN_RAM=1)

target_link_libraries(pico_faces pico_stdlib pico_multicore hardware_vreg
    hardware_clocks)
if(RF_STAGE_DMA)
    target_sources(pico_faces PRIVATE stage.c)
    target_link_libraries(pico_faces hardware_dma)
    target_compile_definitions(pico_faces PRIVATE RF_STAGE_DMA=1)
    if(RF_STAGE_HZ)
        target_compile_definitions(pico_faces PRIVATE RF_STAGE_HZ=${RF_STAGE_HZ})
    endif()
endif()
# RF_VGA=1: scanvideo (IRQ-driven, proven). RF_VGA=2: interrupt-free
# scanout - scanvideo's PIO programs + pins, our static DMA rings
# (firmware/vga_noirq.c); no scanvideo runtime linked.
if(RF_VGA EQUAL 2)
    target_sources(pico_faces PRIVATE vga_noirq.c)
    pico_generate_pio_header(pico_faces
        ${PICO_EXTRAS_PATH}/src/common/pico_scanvideo/scanvideo.pio)
    pico_generate_pio_header(pico_faces
        ${PICO_EXTRAS_PATH}/src/rp2_common/pico_scanvideo_dpi/timing.pio)
    target_include_directories(pico_faces PRIVATE
        ${PICO_EXTRAS_PATH}/src/common/pico_scanvideo/include
        ${PICO_EXTRAS_PATH}/src/rp2_common/pico_scanvideo_dpi/include)
    target_link_libraries(pico_faces hardware_pio hardware_dma)
    target_compile_definitions(pico_faces PRIVATE RF_VGA=2)
    if(RF_VGA_TEST) # boot test pattern through the fb path (debug aid)
        target_compile_definitions(pico_faces PRIVATE RF_VGA_TEST=1)
    endif()
elseif(RF_VGA)
    target_sources(pico_faces PRIVATE vga.c)
    target_link_libraries(pico_faces pico_scanvideo_dpi)
    target_compile_definitions(pico_faces PRIVATE
        RF_VGA=1 PICO_SCANVIDEO_MAX_SCANLINE_BUFFER_WORDS=210)
endif()
pico_enable_stdio_usb(pico_faces 1)
pico_enable_stdio_uart(pico_faces 0)
pico_add_extra_outputs(pico_faces)
