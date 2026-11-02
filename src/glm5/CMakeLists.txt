add_library(strata_glm5_format STATIC geometry.cpp model_file.cpp cache_policy.cpp)
target_include_directories(strata_glm5_format PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(strata_glm5_format PRIVATE strata_warnings)

add_executable(glm5-inspect inspect.cpp)
target_link_libraries(glm5-inspect PRIVATE strata_glm5_format)

if(STRATA_BUILD_TESTS)
  add_executable(glm5_format_test "${PROJECT_SOURCE_DIR}/tests/glm5/format_test.cpp")
  target_include_directories(glm5_format_test PRIVATE "${PROJECT_SOURCE_DIR}/tests/core")
  target_link_libraries(glm5_format_test PRIVATE strata_glm5_format)
  add_test(NAME glm5_format COMMAND glm5_format_test)
  add_executable(glm5_cache_test "${PROJECT_SOURCE_DIR}/tests/glm5/cache_test.cpp")
  target_link_libraries(glm5_cache_test PRIVATE strata_glm5_format)
  add_test(NAME glm5_cache COMMAND glm5_cache_test)
endif()

if(NOT STRATA_ENABLE_CUDA AND NOT STRATA_ENABLE_HIP)
  message(STATUS "GLM5: building the host inspector/tests only; strata-glm5 requires CUDA or HIP")
  return()
endif()
if(STRATA_HIP_GFX906)
  message(FATAL_ERROR "strata-glm5 has not been ported to the gfx906 backend")
endif()
if(NOT STRATA_NATIVE_EXPERTS)
  message(FATAL_ERROR "strata-glm5 requires STRATA_NATIVE_EXPERTS=ON for GGUF expert arithmetic")
endif()

set(glm5_kernels kernels/glm_mhc.cu kernels/glm_kda.cu kernels/glm_delta.cu
                 kernels/glm_attn.cu kernels/glm_dsa.cu kernels/glm_elt.cu)
if(STRATA_ENABLE_HIP)
  set_source_files_properties(${glm5_kernels} PROPERTIES LANGUAGE HIP)
endif()
add_library(strata_glm5_kernels STATIC ${glm5_kernels})
target_include_directories(strata_glm5_kernels PUBLIC "${PROJECT_SOURCE_DIR}/include"
                           ${_strata_gpu_include_directories})
target_link_libraries(strata_glm5_kernels PUBLIC ${_strata_gpu_runtime_target})
target_compile_options(strata_glm5_kernels PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>"
  "$<$<COMPILE_LANGUAGE:HIP>:-ffast-math>")

add_library(strata_glm5_runtime STATIC model.cpp layer.cpp gemv_util.cpp gpu_experts.cpp)
target_include_directories(strata_glm5_runtime PRIVATE ${_strata_gpu_include_directories})
target_link_libraries(strata_glm5_runtime PUBLIC strata_glm5_format strata_glm5_kernels
                      strata_engine)
add_executable(strata-glm5 main.cpp)
target_link_libraries(strata-glm5 PRIVATE strata_glm5_runtime)
strata_utf8_manifest(strata-glm5)

if(STRATA_BUILD_TESTS)
  add_executable(glm5_parity "${PROJECT_SOURCE_DIR}/tests/glm5/parity.cpp")
  target_link_libraries(glm5_parity PRIVATE strata_glm5_kernels)
  add_test(NAME glm5_parity COMMAND glm5_parity --selftest)
  add_executable(glm5_head_test "${PROJECT_SOURCE_DIR}/tests/glm5/head_test.cpp")
  target_link_libraries(glm5_head_test PRIVATE strata_glm5_kernels)
  add_test(NAME glm5_head COMMAND glm5_head_test)
  set_tests_properties(glm5_head PROPERTIES SKIP_RETURN_CODE 77)
  add_executable(glm5_runtime_test "${PROJECT_SOURCE_DIR}/tests/glm5/runtime_test.cpp")
  target_include_directories(glm5_runtime_test PRIVATE "${PROJECT_SOURCE_DIR}/tests/core"
                             ${_strata_gpu_include_directories})
  target_link_libraries(glm5_runtime_test PRIVATE strata_glm5_runtime)
  add_test(NAME glm5_runtime COMMAND glm5_runtime_test)
  set_tests_properties(glm5_runtime PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
endif()
