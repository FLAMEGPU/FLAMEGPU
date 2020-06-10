<?xml version="1.0" encoding="utf-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" 
                xmlns:xmml="http://www.dcs.shef.ac.uk/~paul/XMML"
                xmlns:gpu="http://www.dcs.shef.ac.uk/~paul/XMMLGPU">
<xsl:output method="text" version="1.0" encoding="UTF-8" indent="yes" />
<xsl:include href = "./_common_templates.xslt" />
<xsl:template match="/">
<xsl:call-template name="copyrightNotice"></xsl:call-template>

  //Disable internal thrust warnings about conversions
  #ifdef _MSC_VER
  #pragma warning(push)
  #pragma warning (disable : 4267)
  #pragma warning (disable : 4244)
  #endif
  #ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wunused-parameter"
  #endif

  // includes
  #include &lt;cuda_runtime.h&gt;
#include &lt;device_launch_parameters.h&gt;
#include &lt;stdlib.h&gt;
#include &lt;stdio.h&gt;
#include &lt;string.h&gt;
#include &lt;cmath&gt;
#include &lt;thrust/device_ptr.h&gt;
#include &lt;thrust/scan.h&gt;
#include &lt;thrust/sort.h&gt;
#include &lt;thrust/extrema.h&gt;
#include &lt;thrust/system/cuda/execution_policy.h&gt;
#include &lt;cub/cub.cuh&gt;

// include FLAME kernels
#include "FLAMEGPU_kernals.cu"
<!--Compile time errors for spatial partitioning -->
<xsl:for-each select="gpu:xmodel/xmml:messages/gpu:message/gpu:partitioningSpatial">
<!-- Calculate some values. -->
<xsl:variable name="message_name" select="../xmml:name"/>
<!-- Number of bins in a given dimension is the range divided by the radius, rounded to an integer. Round is used in place of ceil or floor to account for odd floating point values, i.e 12.000...1 should only use 12 bins -->
<xsl:variable name="x_dim"><xsl:value-of select="round((gpu:xmax - gpu:xmin) div gpu:radius)"/></xsl:variable>
<xsl:variable name="y_dim"><xsl:value-of select="round((gpu:ymax - gpu:ymin) div gpu:radius)"/></xsl:variable>
<xsl:variable name="z_dim"><xsl:value-of select="round((gpu:zmax - gpu:zmin) div gpu:radius)"/></xsl:variable>
<!-- If radius is not a factor of the partitioning dimensions as this causes partitioning to execute incorrectly. Check using an epsilon equals xslt template for floating point noise -->
<xsl:variable name="valid_x_dim_factor">
  <xsl:call-template name="epsilonEquals">
    <xsl:with-param name="left" select="gpu:xmax - gpu:xmin"/>
    <xsl:with-param name="right" select="$x_dim * gpu:radius"/>
    <xsl:with-param name="epsilon" select="0.0000000001"/>
  </xsl:call-template>
</xsl:variable>
<xsl:if test="$valid_x_dim_factor='false'">
#error "XML model spatial partitioning radius for for message <xsl:value-of select="$message_name" /> must be a factor of partitioning dimensions. Radius: <xsl:value-of select="gpu:radius"/>, Xmin: <xsl:value-of select="gpu:xmin"/>, Xmax: <xsl:value-of select="gpu:xmax"/>, X_bins: <xsl:value-of select="$x_dim"/>
</xsl:if>
<xsl:variable name="valid_y_dim_factor">
  <xsl:call-template name="epsilonEquals">
    <xsl:with-param name="left" select="gpu:ymax - gpu:ymin"/>
    <xsl:with-param name="right" select="$y_dim * gpu:radius"/>
    <xsl:with-param name="epsilon" select="0.0000000001"/>
  </xsl:call-template>
</xsl:variable>
<xsl:if test="$valid_y_dim_factor='false'">
#error "XML model spatial partitioning radius for for message <xsl:value-of select="$message_name" /> must be a factor of partitioning dimensions. Radius: <xsl:value-of select="gpu:radius"/>, Ymin: <xsl:value-of select="gpu:ymin"/>, Ymax: <xsl:value-of select="gpu:ymax"/>, Y_bins: <xsl:value-of select="$y_dim"/>"
</xsl:if>
<xsl:variable name="valid_z_dim_factor">
  <xsl:call-template name="epsilonEquals">
    <xsl:with-param name="left" select="gpu:zmax - gpu:zmin"/>
    <xsl:with-param name="right" select="$z_dim * gpu:radius"/>
    <xsl:with-param name="epsilon" select="0.0000000001"/>
  </xsl:call-template>
</xsl:variable>
<xsl:if test="$valid_z_dim_factor='false'">
#error "XML model spatial partitioning radius for for message <xsl:value-of select="$message_name" /> must be a factor of partitioning dimensions. Radius: <xsl:value-of select="gpu:radius"/>, Zmin: <xsl:value-of select="gpu:zmin"/>, Zmax: <xsl:value-of select="gpu:zmax"/>, Z_bins: <xsl:value-of select="$z_dim"/>"
</xsl:if>

<!-- If the resulting number of bins in the X or Y planes is less than 3, generate a compile time error. -->
<xsl:if test="$x_dim &lt; 3">
#error "XML model spatial partitioning radius for for message <xsl:value-of select="$message_name" /> is too large for X dimension. ceil((Xmax-Xmin)/Radius) = <xsl:value-of select="$x_dim"/> but must be &gt;= 3. Radius: <xsl:value-of select="gpu:radius"/>, Xmin: <xsl:value-of select="gpu:xmin"/>, Xmax: <xsl:value-of select="gpu:xmax"/>. Consider using partitioningNone."
</xsl:if>
<xsl:if test="$y_dim &lt; 3">
#error "XML model spatial partitioning radius for for message <xsl:value-of select="$message_name" /> is too large for Y dimension. ceil((Xmax-Xmin)/Radius) = <xsl:value-of select="$y_dim"/> but must be &gt;= 3. Radius: <xsl:value-of select="gpu:radius"/>, Ymin: <xsl:value-of select="gpu:ymin"/>, Ymax: <xsl:value-of select="gpu:ymax"/>. Consider using partitioningNone."
</xsl:if>
</xsl:for-each>


<!--Compile time errors for discrete partitioning -->
<xsl:for-each select="gpu:xmodel/xmml:messages/gpu:message/gpu:partitioningDiscrete">
<!-- Calculate some values -->
<xsl:variable name="message_name" select="../xmml:name"/>
<xsl:variable name="bufferSize" select="../gpu:bufferSize"/>
<xsl:variable name="radius" select="gpu:radius"/>
<xsl:variable name="min_buf_for_radius"><xsl:value-of select="(4 * $radius * $radius) + (4 * $radius) + 1"/></xsl:variable>

<!-- If discrete partitioning radius is negative, error -->
<xsl:if test="($radius &lt; 0)">
#error "XML model discrete partitioning radius for message <xsl:value-of select="$message_name" /> must be >= 0"
</xsl:if>

<!-- if 0 or greater, check for other errors-->
<xsl:if test="not($radius &lt; 0)">
<!-- If discrete partitioning radius is too large for the grid error.
This is when (2 * radius) + 1 > grid_width, which can also be expressed as (4r^2 + 4r + 1) > bufferSize -->
<xsl:if test="($bufferSize &lt; $min_buf_for_radius)">
#error "XML model discrete partitioning radius for message <xsl:value-of select="$message_name" /> is too large for bufferSize. Radius must be &lt;= sqrt(bufferSize). bufferSize <xsl:value-of select="$bufferSize" />, Radius: <xsl:value-of select="$radius" />, Minimum bufferSize for radius: <xsl:value-of select="$min_buf_for_radius" />"
</xsl:if>
</xsl:if>
</xsl:for-each>

<!-- Compile time error if there are any discrete agent functions with function conditions -->
<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
<xsl:variable name="agent_name" select="xmml:name"/>
<xsl:if test="gpu:type='discrete'">
<xsl:for-each select="xmml:functions/gpu:function">
<xsl:variable name="function_name" select="xmml:name"/>
<xsl:if test="xmml:condition">
#error "Discrete agent `<xsl:value-of select="$agent_name"/>` cannot have conditional agent function `<xsl:value-of select="$function_name"/>`"
</xsl:if>
</xsl:for-each>
</xsl:if>
</xsl:for-each>

<!-- Compile time errors based on message partitioning and agent types-->
<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent"><xsl:variable name="agent_name" select="xmml:name"/><xsl:variable name="agent_type" select="gpu:type"/>
<xsl:for-each select="xmml:functions/gpu:function"><xsl:variable name="function_name" select="xmml:name"/>
<xsl:for-each select="xmml:outputs/gpu:output"><xsl:variable name="message_name" select="xmml:messageName"/>
<xsl:for-each select="../../../../../../xmml:messages/gpu:message[xmml:name=$message_name]">
<!-- Discrete agents can only output discrete messages -->
<xsl:if test="$agent_type='discrete' and not(gpu:partitioningDiscrete)">
#error "Discrete agent `<xsl:value-of select="$agent_name"/>` can only output partitioningDiscrete messages. `<xsl:value-of select="$message_name"/>` output by `<xsl:value-of select="$function_name"/>` are not partitioningDiscrete. "
</xsl:if>
<!-- Continous agents cannot output discrete messages -->
<xsl:if test="$agent_type='continuous' and gpu:partitioningDiscrete">
#error "Continuous agent `<xsl:value-of select="$agent_name"/>` cannot output partitioningDiscrete messages. `<xsl:value-of select="$message_name"/>` output by `<xsl:value-of select="$function_name"/>` are partitioningDiscrete. "
</xsl:if>
</xsl:for-each>
</xsl:for-each>
</xsl:for-each>
</xsl:for-each>

<!--Compile time error if there are any messages with vector type variables or invalid default values-->
<xsl:for-each select="gpu:xmodel/xmml:messages/gpu:message/xmml:variables/gpu:variable">
<xsl:variable name="message_name" select="../../xmml:name"/>
<xsl:variable name="variable_name" select="xmml:name"/>
<xsl:variable name="variable_type" select="xmml:type"/>
<xsl:variable name="defaultValue" select="xmml:defaultValue" />

<!-- check for invalid defaultValues for scalar message variables -->
<xsl:if test="$defaultValue and not(contains($variable_type, 'vec'))">
<xsl:variable name="numValues" select="1" />
<xsl:variable name="expectedCommas" select="$numValues - 1" />
<xsl:variable name="numCommas" select="string-length($defaultValue) - string-length(translate($defaultValue, ',', ''))" />
<xsl:if test="not($numCommas=$expectedCommas)">
#error "Invalid defaultValue of `<xsl:value-of select="$defaultValue" />` for message `<xsl:value-of select="$message_name" />` variable `<xsl:value-of select="$variable_name" />`. `<xsl:value-of select="$variable_type" />` requires a single value"
</xsl:if>
</xsl:if>
<!-- check for vector type message variables -->
<xsl:if test="contains($variable_type, 'vec')">
#error "Message `<xsl:value-of select="$message_name" />` contains vector type message variable `<xsl:value-of select="$variable_name" />` of type `<xsl:value-of select="$variable_type" />`"
</xsl:if>
</xsl:for-each>

<!-- Compile time error for any incorrect default values. -->
<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:memory/gpu:variable">
<xsl:if test="xmml:defaultValue">
<xsl:variable name="agent_name" select="../../xmml:name"/>
<xsl:variable name="variable_name" select="xmml:name"/>
<xsl:variable name="variable_type" select="xmml:type"/>
<xsl:variable name="defaultValue" select="xmml:defaultValue" />
<xsl:variable name="numCommas" select="string-length($defaultValue) - string-length(translate($defaultValue, ',', ''))" />
<!-- Non vectors require no commas -->
<xsl:if test="not(contains($variable_type, 'vec'))">
<xsl:variable name="numValues" select="1" />
<xsl:variable name="expectedCommas" select="$numValues - 1" />
<xsl:if test="not($numCommas=$expectedCommas)">
#error "Invalid defaultValue of `<xsl:value-of select="$defaultValue" />` for xagent `<xsl:value-of select="$agent_name" />` variable `<xsl:value-of select="$variable_name" />`. `<xsl:value-of select="$variable_type" />` requires a single value"
</xsl:if>
</xsl:if>
<!-- vector types require an appropriate number of commas -->
<xsl:if test="contains($variable_type, 'vec')">
<xsl:variable name="numValues" select="substring($variable_type, string-length($variable_type))" />
<xsl:variable name="expectedCommas" select="$numValues - 1" />
<xsl:if test="not($numCommas=$expectedCommas)">
#error "Invalid defaultValue of `<xsl:value-of select="$defaultValue" />` for xagent `<xsl:value-of select="$agent_name" />` variable `<xsl:value-of select="$variable_name" />`. `<xsl:value-of select="$variable_type" />` requires <xsl:value-of select="$numValues" /> comma separated values"
</xsl:if>
</xsl:if>
</xsl:if>
</xsl:for-each>

#ifdef _MSC_VER
#pragma warning(pop)
#endif
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

/* Error check function for safe CUDA API calling */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* Error check function for post CUDA Kernel calling */
#define gpuErrchkLaunch() { gpuLaunchAssert(__FILE__, __LINE__); }
inline void gpuLaunchAssert(const char *file, int line, bool abort=true)
{
	gpuAssert( cudaPeekAtLastError(), file, line );
#ifdef _DEBUG
	gpuAssert( cudaDeviceSynchronize(), file, line );
#endif
   
}

/* SM padding and offset variables */
int SM_START;
int PADDING;

unsigned int g_iterationNumber;

/* Agent Memory */
<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
/* <xsl:value-of select="xmml:name"/> Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_<xsl:value-of select="xmml:name"/>_list* d_<xsl:value-of select="xmml:name"/>s;      /**&lt; Pointer to agent list (population) on the device*/
xmachine_memory_<xsl:value-of select="xmml:name"/>_list* d_<xsl:value-of select="xmml:name"/>s_swap; /**&lt; Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_<xsl:value-of select="xmml:name"/>_list* d_<xsl:value-of select="xmml:name"/>s_new;  /**&lt; Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_<xsl:value-of select="xmml:name"/>_count;   /**&lt; Agent population size counter */ <xsl:if test="gpu:type='discrete'">
int h_xmachine_memory_<xsl:value-of select="xmml:name"/>_pop_width;   /**&lt; Agent population width */</xsl:if>
uint * d_xmachine_memory_<xsl:value-of select="xmml:name"/>_keys;	  /**&lt; Agent sort identifiers keys*/
uint * d_xmachine_memory_<xsl:value-of select="xmml:name"/>_values;  /**&lt; Agent sort identifiers value */
<xsl:for-each select="xmml:states/gpu:state">
/* <xsl:value-of select="../../xmml:name"/> state variables */
xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>;      /**&lt; Pointer to agent list (population) on host*/
xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>;      /**&lt; Pointer to agent list (population) on the device*/
int h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count;   /**&lt; Agent population size counter */ 
</xsl:for-each>
</xsl:for-each>

/* Variables to track the state of host copies of state lists, for the purposes of host agent data access.
 * @future - if the host data is current it may be possible to avoid duplicating memcpy in xml output.
 */
<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent"><xsl:variable name="agent_name" select="xmml:name"/>
<xsl:for-each select="xmml:states/gpu:state"><xsl:variable name="agent_state" select="xmml:name"/>
<xsl:for-each select="../../xmml:memory/gpu:variable"><xsl:variable name="variable_name" select="xmml:name"/><xsl:variable name="variable_type" select="xmml:type" />unsigned int h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$agent_state"/>_variable_<xsl:value-of select="$variable_name"/>_data_iteration;
</xsl:for-each>
</xsl:for-each>
</xsl:for-each>

/* Message Memory */
<xsl:for-each select="gpu:xmodel/xmml:messages/gpu:message">
/* <xsl:value-of select="xmml:name"/> Message variables */
xmachine_message_<xsl:value-of select="xmml:name"/>_list* h_<xsl:value-of select="xmml:name"/>s;         /**&lt; Pointer to message list on host*/
xmachine_message_<xsl:value-of select="xmml:name"/>_list* d_<xsl:value-of select="xmml:name"/>s;         /**&lt; Pointer to message list on device*/
xmachine_message_<xsl:value-of select="xmml:name"/>_list* d_<xsl:value-of select="xmml:name"/>s_swap;    /**&lt; Pointer to message swap list on device (used for holding optional messages)*/
<xsl:if test="gpu:partitioningNone or gpu:partitioningSpatial">/* Non partitioned and spatial partitioned message variables  */
int h_message_<xsl:value-of select="xmml:name"/>_count;         /**&lt; message list counter*/
int h_message_<xsl:value-of select="xmml:name"/>_output_type;   /**&lt; message output type (single or optional)*/
</xsl:if>
<xsl:if test="gpu:partitioningGraphEdge">/* On-Graph Partitioned message variables  */
unsigned int h_message_<xsl:value-of select="xmml:name"/>_count;         /**&lt; message list counter*/
int h_message_<xsl:value-of select="xmml:name"/>_output_type;   /**&lt; message output type (single or optional)*/
</xsl:if>
<xsl:if test="gpu:partitioningSpatial">/* Spatial Partitioning Variables*/
#ifdef FAST_ATOMIC_SORTING
	uint * d_xmachine_message_<xsl:value-of select="xmml:name"/>_local_bin_index;	  /**&lt; index offset within the assigned bin */
	uint * d_xmachine_message_<xsl:value-of select="xmml:name"/>_unsorted_index;		/**&lt; unsorted index (hash) value for message */
    // Values for CUB exclusive scan of spatially partitioned variables
    void * d_temp_scan_storage_xmachine_message_<xsl:value-of select="xmml:name" />;
    size_t temp_scan_bytes_xmachine_message_<xsl:value-of select="xmml:name" />;
#else
	uint * d_xmachine_message_<xsl:value-of select="xmml:name"/>_keys;	  /**&lt; message sort identifier keys*/
	uint * d_xmachine_message_<xsl:value-of select="xmml:name"/>_values;  /**&lt; message sort identifier values */
  uint * d_xmachine_message_<xsl:value-of select="xmml:name"/>_keys_swap;	  /**&lt; message sort identifier keys*/
  uint * d_xmachine_message_<xsl:value-of select="xmml:name"/>_values_swap;  /**&lt; message sort identifier values */

  size_t CUB_temp_storage_bytes_<xsl:value-of select="xmml:name"/> = 0;
  void *d_CUB_temp_storage_<xsl:value-of select="xmml:name"/> = nullptr;
  const unsigned int binCountBits_<xsl:value-of select="xmml:name"/> = (unsigned int)ceil(log(xmachine_message_<xsl:value-of select="xmml:name"/>_grid_size) / log(2));
  #endif
  xmachine_message_<xsl:value-of select="xmml:name"/>_PBM * d_<xsl:value-of select="xmml:name"/>_partition_matrix;  /**&lt; Pointer to PCB matrix */
glm::vec3 h_message_<xsl:value-of select="xmml:name"/>_min_bounds;           /**&lt; min bounds (x,y,z) of partitioning environment */
glm::vec3 h_message_<xsl:value-of select="xmml:name"/>_max_bounds;           /**&lt; max bounds (x,y,z) of partitioning environment */
glm::ivec3 h_message_<xsl:value-of select="xmml:name"/>_partitionDim;           /**&lt; partition dimensions (x,y,z) of partitioning environment */
float h_message_<xsl:value-of select="xmml:name"/>_radius;                 /**&lt; partition radius (used to determin the size of the partitions) */
</xsl:if><xsl:if test="gpu:partitioningDiscrete">/* Discrete Partitioning Variables*/
int h_message_<xsl:value-of select="xmml:name"/>_range;     /**&lt; range of the discrete message*/
int h_message_<xsl:value-of select="xmml:name"/>_width;     /**&lt; with of the message grid*/
</xsl:if><xsl:if test="gpu:partitioningDiscrete or gpu:partitioningSpatial">/* Texture offset values for host */<xsl:for-each select="xmml:variables/gpu:variable">
int h_tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_offset;</xsl:for-each>
<xsl:if test="gpu:partitioningSpatial">
int h_tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_start_offset;
int h_tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_end_or_count_offset;
</xsl:if></xsl:if>
<xsl:if test="gpu:partitioningGraphEdge">/* On-Graph Partitioning Variables */
// Message bounds structure
xmachine_message_<xsl:value-of select="xmml:name"/>_bounds * d_xmachine_message_<xsl:value-of select="xmml:name"/>_bounds;
// Temporary data used during the scattering of messages
xmachine_message_<xsl:value-of select="xmml:name"/>_scatterer * d_xmachine_message_<xsl:value-of select="xmml:name"/>_scatterer; 
// Values for CUB exclusive scan of spatially partitioned variables
void * d_temp_scan_storage_xmachine_message_<xsl:value-of select="xmml:name" />;
size_t temp_scan_bytes_xmachine_message_<xsl:value-of select="xmml:name" />;
</xsl:if>
</xsl:for-each>
  
/* CUDA Streams for function layers */<xsl:for-each select="gpu:xmodel/xmml:layers/xmml:layer">
<xsl:sort select="count(gpu:layerFunction)" order="descending"/>
<xsl:if test="position() =1"> <!-- Get the layer with most functions -->
<xsl:for-each select="gpu:layerFunction">
cudaStream_t stream<xsl:value-of select="position()"/>;</xsl:for-each>
</xsl:if>
</xsl:for-each>

/* Device memory and sizes for CUB values */
<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
void * d_temp_scan_storage_<xsl:value-of select="xmml:name" />;
size_t temp_scan_storage_bytes_<xsl:value-of select="xmml:name" />;
</xsl:for-each>

/*Global condition counts*/<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:functions/gpu:function/gpu:globalCondition">
int h_<xsl:value-of select="../xmml:name"/>_condition_count;
</xsl:for-each>

/* Agent ID Generation functions implemented in simulation.cu and FLAMEGPU_kernals.cu*/
<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
<xsl:variable name="agent_name" select="xmml:name" />
<xsl:for-each select="xmml:memory/gpu:variable">
<xsl:variable name="variable_name" select="xmml:name" />
<xsl:variable name="variable_type" select="xmml:type" />
<xsl:variable name="type_is_integer"><xsl:call-template name="typeIsInteger"><xsl:with-param name="type" select="$variable_type"/></xsl:call-template></xsl:variable>
<!-- If the agent has a variable name id, of a single integer type -->
<xsl:if test="$variable_name='id' and not(xmml:arrayLength) and $type_is_integer='true'" >
<xsl:value-of select="$variable_type"/> h_current_value_generate_<xsl:value-of select="$agent_name"/>_id = 0;

// Track the last value returned from the device, to enable copying to the device after a step function.
<xsl:value-of select="$variable_type"/> h_last_value_generate_<xsl:value-of select="$agent_name"/>_id = <xsl:call-template name="maximumIntegerValue"><xsl:with-param name="type" select="$variable_type"/></xsl:call-template>;

void set_initial_<xsl:value-of select="$agent_name"/>_id(<xsl:value-of select="$variable_type" /> firstID){
  h_current_value_generate_<xsl:value-of select="$agent_name"/>_id = firstID;
}

// Function to copy from the host to the device in the default stream
void update_device_generate_<xsl:value-of select="$agent_name"/>_id(){
// If the last device value doesn't match the current value, update the device value. 
  if(h_current_value_generate_<xsl:value-of select="$agent_name"/>_id != h_last_value_generate_<xsl:value-of select="$agent_name"/>_id){
    gpuErrchk(cudaMemcpyToSymbol( d_current_value_generate_<xsl:value-of select="$agent_name"/>_id, &amp;h_current_value_generate_<xsl:value-of select="$agent_name"/>_id, sizeof(<xsl:value-of select="$variable_type" />)));
  }
}
// Function to copy from the device to the host in the default stream
void update_host_generate_<xsl:value-of select="$agent_name"/>_id(){
  gpuErrchk(cudaMemcpyFromSymbol( &amp;h_current_value_generate_<xsl:value-of select="$agent_name"/>_id, d_current_value_generate_<xsl:value-of select="$agent_name"/>_id, sizeof(<xsl:value-of select="$variable_type" />)));
  h_last_value_generate_<xsl:value-of select="$agent_name"/>_id = h_current_value_generate_<xsl:value-of select="$agent_name"/>_id;
}
</xsl:if>
</xsl:for-each>
</xsl:for-each>


/* RNG rand48 */
RNG_rand48* h_rand48;    /**&lt; Pointer to RNG_rand48 seed list on host*/
RNG_rand48* d_rand48;    /**&lt; Pointer to RNG_rand48 seed list on device*/

/* Early simulation exit*/
bool g_exit_early;

/* Cuda Event Timers for Instrumentation */
#if defined(INSTRUMENT_ITERATIONS) &amp;&amp; INSTRUMENT_ITERATIONS
	cudaEvent_t instrument_iteration_start, instrument_iteration_stop;
	float instrument_iteration_milliseconds = 0.0f;
#endif
#if (defined(INSTRUMENT_AGENT_FUNCTIONS) &amp;&amp; INSTRUMENT_AGENT_FUNCTIONS) || (defined(INSTRUMENT_INIT_FUNCTIONS) &amp;&amp; INSTRUMENT_INIT_FUNCTIONS) || (defined(INSTRUMENT_STEP_FUNCTIONS) &amp;&amp; INSTRUMENT_STEP_FUNCTIONS) || (defined(INSTRUMENT_EXIT_FUNCTIONS) &amp;&amp; INSTRUMENT_EXIT_FUNCTIONS)
	cudaEvent_t instrument_start, instrument_stop;
	float instrument_milliseconds = 0.0f;
#endif

/* CUDA Parallel Primatives variables */
int scan_last_sum;           /**&lt; Indicates if the position (in message list) of last message*/
int scan_last_included;      /**&lt; Indicates if last sum value is included in the total sum count*/

/* Agent function prototypes */
<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:functions/gpu:function">
/** <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>
 * Agent function prototype for <xsl:value-of select="xmml:name"/> function of <xsl:value-of select="../../xmml:name"/> agent
 */
void <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>(cudaStream_t &amp;stream);
</xsl:for-each>
  
void setPaddingAndOffset()
{
    PROFILE_SCOPED_RANGE("setPaddingAndOffset");
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&amp;deviceProp, 0);
	int x64_sys = 0;

	// This function call returns 9999 for both major &amp; minor fields, if no CUDA capable devices are present
	if (deviceProp.major == 9999 &amp;&amp; deviceProp.minor == 9999){
		printf("Error: There is no device supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}
    
    //check if double is used and supported
#ifdef _DOUBLE_SUPPORT_REQUIRED_
	printf("Simulation requires full precision double values\n");
	if ((deviceProp.major &lt; 2)&amp;&amp;(deviceProp.minor &lt; 3)){
		printf("Error: Hardware does not support full precision double values!\n");
		exit(EXIT_FAILURE);
	}
    
#endif

	//check 32 or 64bit
	x64_sys = (sizeof(void*)==8);
	if (x64_sys)
	{
		printf("64Bit System Detected\n");
	}
	else
	{
		printf("32Bit System Detected\n");
	}

	SM_START = 0;
	PADDING = 0;
  
	//copy padding and offset to GPU
	gpuErrchk(cudaMemcpyToSymbol( d_SM_START, &amp;SM_START, sizeof(int)));
	gpuErrchk(cudaMemcpyToSymbol( d_PADDING, &amp;PADDING, sizeof(int)));     
}

int is_sqr_pow2(int x){
	int r = (int)pow(4, ceil(log(x)/log(4)));
	return (r == x);
}

int lowest_sqr_pow2(int x){
	int l;
	
	//escape early if x is square power of 2
	if (is_sqr_pow2(x))
		return x;
	
	//lower bound		
	l = (int)pow(4, floor(log(x)/log(4)));
	
	return l;
}

/* Unary function required for cudaOccupancyMaxPotentialBlockSizeVariableSMem to avoid warnings */
int no_sm(int b){
	return 0;
}

/* Unary function to return shared memory size for reorder message kernels */
int reorder_messages_sm_size(int blockSize)
{
	return sizeof(unsigned int)*(blockSize+1);
}


/** getIterationNumber
 *  Get the iteration number (host)
 *  @return a 1 indexed value for the iteration number, which is incremented at the start of each simulation step.
 *      I.e. it is 0 on up until the first call to singleIteration()
 */
extern unsigned int getIterationNumber(){
    return g_iterationNumber;
}

void initialise(char * inputfile){
    PROFILE_SCOPED_RANGE("initialise");

	//set the padding and offset values depending on architecture and OS
	setPaddingAndOffset();
  
		// Initialise some global variables
		g_iterationNumber = 0;
		g_exit_early = false;

    // Initialise variables for tracking which iterations' data is accessible on the host.
    <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent"><xsl:variable name="agent_name" select="xmml:name"/><xsl:for-each select="xmml:states/gpu:state"><xsl:variable name="agent_state" select="xmml:name"/><xsl:for-each select="../../xmml:memory/gpu:variable"><xsl:variable name="variable_name" select="xmml:name"/><xsl:variable name="variable_type" select="xmml:type" />h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$agent_state"/>_variable_<xsl:value-of select="$variable_name"/>_data_iteration = 0;
    </xsl:for-each></xsl:for-each></xsl:for-each>



	printf("Allocating Host and Device memory\n");
    PROFILE_PUSH_RANGE("allocate host");
	/* Agent memory allocation (CPU) */<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
	int xmachine_<xsl:value-of select="xmml:name"/>_SoA_size = sizeof(xmachine_memory_<xsl:value-of select="xmml:name"/>_list);<xsl:for-each select="xmml:states/gpu:state">
	h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/> = (xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list*)malloc(xmachine_<xsl:value-of select="../../xmml:name"/>_SoA_size);</xsl:for-each></xsl:for-each>

	/* Message memory allocation (CPU) */<xsl:for-each select="gpu:xmodel/xmml:messages/gpu:message">
	int message_<xsl:value-of select="xmml:name"/>_SoA_size = sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>_list);
	h_<xsl:value-of select="xmml:name"/>s = (xmachine_message_<xsl:value-of select="xmml:name"/>_list*)malloc(message_<xsl:value-of select="xmml:name"/>_SoA_size);</xsl:for-each>

	//Exit if agent or message buffer sizes are to small for function outputs<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:functions/gpu:function/xmml:xagentOutputs/gpu:xagentOutput">
	<xsl:variable name="xagent_output" select="xmml:xagentName"/><xsl:variable name="xagent_buffer" select="../../../../gpu:bufferSize"/><xsl:if test="../../../../../gpu:xagent[xmml:name=$xagent_output]/gpu:bufferSize&lt;$xagent_buffer">
	printf("ERROR: <xsl:value-of select="$xagent_output"/> agent buffer is too small to be used for output by <xsl:value-of select="../../../../xmml:name"/> agent in <xsl:value-of select="../../xmml:name"/> function!\n");
    PROFILE_POP_RANGE(); //"allocate host"
	exit(EXIT_FAILURE);
	</xsl:if>    
	</xsl:for-each>

  /* Graph memory allocation (CPU) */
  <xsl:for-each select="gpu:xmodel/gpu:environment/gpu:graphs/gpu:staticGraph">
    // Allocate host structure used to load data for device copying
    h_staticGraph_memory_<xsl:value-of select="gpu:name"/> = (staticGraph_memory_<xsl:value-of select="gpu:name"/>*) malloc(sizeof(staticGraph_memory_<xsl:value-of select="gpu:name"/>));
    // Ensure allocation was successful.
    if(h_staticGraph_memory_<xsl:value-of select="gpu:name"/> == nullptr ){
        printf("FATAL ERROR: Could not allocate host memory for static network <xsl:value-of select="gpu:name"/> \n");
        PROFILE_POP_RANGE();
        exit(EXIT_FAILURE);
    }
  </xsl:for-each>

    PROFILE_POP_RANGE(); //"allocate host"
	<xsl:for-each select="gpu:xmodel/xmml:messages/gpu:message"><xsl:if test="gpu:partitioningDiscrete">
	
	/* Set discrete <xsl:value-of select="xmml:name"/> message variables (range, width)*/
	h_message_<xsl:value-of select="xmml:name"/>_range = <xsl:value-of select="gpu:partitioningDiscrete/gpu:radius"/>; //from xml
	h_message_<xsl:value-of select="xmml:name"/>_width = (int)floor(sqrt((float)xmachine_message_<xsl:value-of select="xmml:name"/>_MAX));
	//check the width
	if (!is_sqr_pow2(xmachine_message_<xsl:value-of select="xmml:name"/>_MAX)){
		printf("ERROR: <xsl:value-of select="xmml:name"/> message max must be a square power of 2 for a 2D discrete message grid!\n");
		exit(EXIT_FAILURE);
	}
	gpuErrchk(cudaMemcpyToSymbol( d_message_<xsl:value-of select="xmml:name"/>_range, &amp;h_message_<xsl:value-of select="xmml:name"/>_range, sizeof(int)));	
	gpuErrchk(cudaMemcpyToSymbol( d_message_<xsl:value-of select="xmml:name"/>_width, &amp;h_message_<xsl:value-of select="xmml:name"/>_width, sizeof(int)));
	</xsl:if><xsl:if test="gpu:partitioningSpatial">
			
	/* Set spatial partitioning <xsl:value-of select="xmml:name"/> message variables (min_bounds, max_bounds)*/
	h_message_<xsl:value-of select="xmml:name"/>_radius = (float)<xsl:value-of select="gpu:partitioningSpatial/gpu:radius"/>;
	gpuErrchk(cudaMemcpyToSymbol( d_message_<xsl:value-of select="xmml:name"/>_radius, &amp;h_message_<xsl:value-of select="xmml:name"/>_radius, sizeof(float)));	
	    h_message_<xsl:value-of select="xmml:name"/>_min_bounds = glm::vec3((float)<xsl:value-of select="gpu:partitioningSpatial/gpu:xmin"/>, (float)<xsl:value-of select="gpu:partitioningSpatial/gpu:ymin"/>, (float)<xsl:value-of select="gpu:partitioningSpatial/gpu:zmin"/>);
	gpuErrchk(cudaMemcpyToSymbol( d_message_<xsl:value-of select="xmml:name"/>_min_bounds, &amp;h_message_<xsl:value-of select="xmml:name"/>_min_bounds, sizeof(glm::vec3)));	
	h_message_<xsl:value-of select="xmml:name"/>_max_bounds = glm::vec3((float)<xsl:value-of select="gpu:partitioningSpatial/gpu:xmax"/>, (float)<xsl:value-of select="gpu:partitioningSpatial/gpu:ymax"/>, (float)<xsl:value-of select="gpu:partitioningSpatial/gpu:zmax"/>);
	gpuErrchk(cudaMemcpyToSymbol( d_message_<xsl:value-of select="xmml:name"/>_max_bounds, &amp;h_message_<xsl:value-of select="xmml:name"/>_max_bounds, sizeof(glm::vec3)));	
	h_message_<xsl:value-of select="xmml:name"/>_partitionDim.x = (int)round((h_message_<xsl:value-of select="xmml:name"/>_max_bounds.x - h_message_<xsl:value-of select="xmml:name"/>_min_bounds.x)/h_message_<xsl:value-of select="xmml:name"/>_radius);
	h_message_<xsl:value-of select="xmml:name"/>_partitionDim.y = (int)round((h_message_<xsl:value-of select="xmml:name"/>_max_bounds.y - h_message_<xsl:value-of select="xmml:name"/>_min_bounds.y)/h_message_<xsl:value-of select="xmml:name"/>_radius);
	h_message_<xsl:value-of select="xmml:name"/>_partitionDim.z = (int)round((h_message_<xsl:value-of select="xmml:name"/>_max_bounds.z - h_message_<xsl:value-of select="xmml:name"/>_min_bounds.z)/h_message_<xsl:value-of select="xmml:name"/>_radius);
	gpuErrchk(cudaMemcpyToSymbol( d_message_<xsl:value-of select="xmml:name"/>_partitionDim, &amp;h_message_<xsl:value-of select="xmml:name"/>_partitionDim, sizeof(glm::ivec3)));	
	</xsl:if></xsl:for-each>
	
	
	<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent"><xsl:if test="gpu:type='discrete'">
	/* Check that population size is a square power of 2*/
	if (!is_sqr_pow2(xmachine_memory_<xsl:value-of select="xmml:name"/>_MAX)){
		printf("ERROR: <xsl:value-of select="xmml:name"/>s agent count must be a square power of 2!\n");
		exit(EXIT_FAILURE);
	}
	h_xmachine_memory_<xsl:value-of select="xmml:name"/>_pop_width = (int)sqrt(xmachine_memory_<xsl:value-of select="xmml:name"/>_MAX);
	</xsl:if></xsl:for-each>

	//read initial states
	readInitialStates(inputfile, <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">h_<xsl:value-of select="xmml:name"/>s_<xsl:value-of select="xmml:states/xmml:initialState"/>, &amp;h_xmachine_memory_<xsl:value-of select="xmml:name"/>_<xsl:value-of select="xmml:states/xmml:initialState"/>_count<xsl:if test="position()!=last()">, </xsl:if></xsl:for-each>);

  // Read graphs from disk
  <xsl:for-each select="gpu:xmodel/gpu:environment/gpu:graphs/gpu:staticGraph">
  <xsl:if test="gpu:loadFromFile/gpu:json">load_staticGraph_<xsl:value-of select="gpu:name"/>_from_json("<xsl:value-of select="gpu:loadFromFile/gpu:json"/>", h_staticGraph_memory_<xsl:value-of select="gpu:name"/>);
  </xsl:if>
  <xsl:if test="gpu:loadFromFile/gpu:xml">load_staticGraph_<xsl:value-of select="gpu:name"/>_from_xml("<xsl:value-of select="gpu:loadFromFile/gpu:xml"/>", h_staticGraph_memory_<xsl:value-of select="gpu:name"/>);
  </xsl:if>
  </xsl:for-each>

  PROFILE_PUSH_RANGE("allocate device");
	<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
	/* <xsl:value-of select="xmml:name"/> Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &amp;d_<xsl:value-of select="xmml:name"/>s, xmachine_<xsl:value-of select="xmml:name"/>_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &amp;d_<xsl:value-of select="xmml:name"/>s_swap, xmachine_<xsl:value-of select="xmml:name"/>_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &amp;d_<xsl:value-of select="xmml:name"/>s_new, xmachine_<xsl:value-of select="xmml:name"/>_SoA_size));
    <xsl:if test="gpu:type='continuous'">//continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &amp;d_xmachine_memory_<xsl:value-of select="xmml:name"/>_keys, xmachine_memory_<xsl:value-of select="xmml:name"/>_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &amp;d_xmachine_memory_<xsl:value-of select="xmml:name"/>_values, xmachine_memory_<xsl:value-of select="xmml:name"/>_MAX* sizeof(uint)));</xsl:if>
    <xsl:for-each select="xmml:states/gpu:state">
	/* <xsl:value-of select="xmml:name"/> memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &amp;d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, xmachine_<xsl:value-of select="../../xmml:name"/>_SoA_size));
	gpuErrchk( cudaMemcpy( d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, xmachine_<xsl:value-of select="../../xmml:name"/>_SoA_size, cudaMemcpyHostToDevice));
    </xsl:for-each>
	</xsl:for-each>

	<xsl:for-each select="gpu:xmodel/xmml:messages/gpu:message">
	/* <xsl:value-of select="xmml:name"/> Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &amp;d_<xsl:value-of select="xmml:name"/>s, message_<xsl:value-of select="xmml:name"/>_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &amp;d_<xsl:value-of select="xmml:name"/>s_swap, message_<xsl:value-of select="xmml:name"/>_SoA_size));
	gpuErrchk( cudaMemcpy( d_<xsl:value-of select="xmml:name"/>s, h_<xsl:value-of select="xmml:name"/>s, message_<xsl:value-of select="xmml:name"/>_SoA_size, cudaMemcpyHostToDevice));<xsl:if test="gpu:partitioningSpatial">
	gpuErrchk( cudaMalloc( (void**) &amp;d_<xsl:value-of select="xmml:name"/>_partition_matrix, sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>_PBM)));
#ifdef FAST_ATOMIC_SORTING
	gpuErrchk( cudaMalloc( (void**) &amp;d_xmachine_message_<xsl:value-of select="xmml:name"/>_local_bin_index, xmachine_message_<xsl:value-of select="xmml:name"/>_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &amp;d_xmachine_message_<xsl:value-of select="xmml:name"/>_unsorted_index, xmachine_message_<xsl:value-of select="xmml:name"/>_MAX* sizeof(uint)));
    /* Calculate and allocate CUB temporary memory for exclusive scans */
    d_temp_scan_storage_xmachine_message_<xsl:value-of select="xmml:name"/> = nullptr;
    temp_scan_bytes_xmachine_message_<xsl:value-of select="xmml:name"/> = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_xmachine_message_<xsl:value-of select="xmml:name"/>, 
        temp_scan_bytes_xmachine_message_<xsl:value-of select="xmml:name"/>, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_message_<xsl:value-of select="xmml:name"/>_grid_size
    );
    gpuErrchk(cudaMalloc(&amp;d_temp_scan_storage_xmachine_message_<xsl:value-of select="xmml:name"/>, temp_scan_bytes_xmachine_message_<xsl:value-of select="xmml:name"/>));
#else
	gpuErrchk( cudaMalloc( (void**) &amp;d_xmachine_message_<xsl:value-of select="xmml:name"/>_keys, xmachine_message_<xsl:value-of select="xmml:name"/>_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &amp;d_xmachine_message_<xsl:value-of select="xmml:name"/>_values, xmachine_message_<xsl:value-of select="xmml:name"/>_MAX* sizeof(uint)));
    gpuErrchk( cudaMalloc( (void**) &amp;d_xmachine_message_<xsl:value-of select="xmml:name"/>_keys_swap, xmachine_message_<xsl:value-of select="xmml:name"/>_MAX* sizeof(uint)));
    gpuErrchk( cudaMalloc( (void**) &amp;d_xmachine_message_<xsl:value-of select="xmml:name"/>_values_swap, xmachine_message_<xsl:value-of select="xmml:name"/>_MAX* sizeof(uint)));
    cub::DeviceRadixSort::SortPairs(d_CUB_temp_storage_<xsl:value-of select="xmml:name"/>, CUB_temp_storage_bytes_<xsl:value-of select="xmml:name"/>, d_xmachine_message_<xsl:value-of select="xmml:name"/>_keys, d_xmachine_message_<xsl:value-of select="xmml:name"/>_keys_swap, d_xmachine_message_<xsl:value-of select="xmml:name"/>_values, d_xmachine_message_<xsl:value-of select="xmml:name"/>_values_swap, xmachine_message_<xsl:value-of select="xmml:name"/>_MAX, 0, binCountBits_<xsl:value-of select="xmml:name"/>);
    gpuErrchk(cudaMalloc((void**)&amp;d_CUB_temp_storage_<xsl:value-of select="xmml:name"/>, CUB_temp_storage_bytes_<xsl:value-of select="xmml:name"/>));
#endif</xsl:if><xsl:if test="gpu:partitioningGraphEdge">
  gpuErrchk(cudaMalloc((void**)&amp;d_xmachine_message_<xsl:value-of select="xmml:name"/>_bounds, sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>_bounds)));
  gpuErrchk(cudaMalloc((void**)&amp;d_xmachine_message_<xsl:value-of select="xmml:name"/>_scatterer, sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>_scatterer)));
  /* Calculate and allocate CUB temporary memory for exclusive scans */
    d_temp_scan_storage_xmachine_message_<xsl:value-of select="xmml:name"/> = nullptr;
    temp_scan_bytes_xmachine_message_<xsl:value-of select="xmml:name"/> = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_xmachine_message_<xsl:value-of select="xmml:name"/>, 
        temp_scan_bytes_xmachine_message_<xsl:value-of select="xmml:name"/>, 
        (unsigned int*) nullptr, 
        (unsigned int*) nullptr, 
        staticGraph_<xsl:value-of select="gpu:partitioningGraphEdge/gpu:environmentGraph"/>_edge_bufferSize
    );
    gpuErrchk(cudaMalloc(&amp;d_temp_scan_storage_xmachine_message_<xsl:value-of select="xmml:name"/>, temp_scan_bytes_xmachine_message_<xsl:value-of select="xmml:name"/>));
  </xsl:if><xsl:text>
	</xsl:text></xsl:for-each>	


  /* Allocate device memory for graphs */
  <xsl:for-each select="gpu:xmodel/gpu:environment/gpu:graphs/gpu:staticGraph">
  // Allocate device memory, this is freed by cleanup() in simulation.cu
  gpuErrchk(cudaMalloc((void**)&amp;d_staticGraph_memory_<xsl:value-of select="gpu:name"/>, sizeof(staticGraph_memory_<xsl:value-of select="gpu:name"/>)));

  // Copy data to the Device
  gpuErrchk(cudaMemcpy(d_staticGraph_memory_<xsl:value-of select="gpu:name"/>, h_staticGraph_memory_<xsl:value-of select="gpu:name"/>, sizeof(staticGraph_memory_<xsl:value-of select="gpu:name"/>), cudaMemcpyHostToDevice));

  // Copy device pointer(s) to CUDA constant(s)
  gpuErrchk(cudaMemcpyToSymbol(d_staticGraph_memory_<xsl:value-of select="gpu:name"/>_ptr, &amp;d_staticGraph_memory_<xsl:value-of select="gpu:name"/>, sizeof(staticGraph_memory_<xsl:value-of select="gpu:name"/>*)));
  </xsl:for-each>

    PROFILE_POP_RANGE(); // "allocate device"

    /* Calculate and allocate CUB temporary memory for exclusive scans */
    <!-- @optimisation only do this for agents which require cub scan memory -->
    <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
    d_temp_scan_storage_<xsl:value-of select="xmml:name"/> = nullptr;
    temp_scan_storage_bytes_<xsl:value-of select="xmml:name"/> = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_<xsl:value-of select="xmml:name"/>, 
        temp_scan_storage_bytes_<xsl:value-of select="xmml:name"/>, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_<xsl:value-of select="xmml:name"/>_MAX
    );
    gpuErrchk(cudaMalloc(&amp;d_temp_scan_storage_<xsl:value-of select="xmml:name"/>, temp_scan_storage_bytes_<xsl:value-of select="xmml:name"/>));
    </xsl:for-each>

	/*Set global condition counts*/<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:functions/gpu:function/gpu:condition">
	h_<xsl:value-of select="../xmml:name"/>_condition_false_count = 0;
	</xsl:for-each>

	/* RNG rand48 */
    PROFILE_PUSH_RANGE("Initialse RNG_rand48");
	int h_rand48_SoA_size = sizeof(RNG_rand48);
	h_rand48 = (RNG_rand48*)malloc(h_rand48_SoA_size);
	//allocate on GPU
	gpuErrchk( cudaMalloc( (void**) &amp;d_rand48, h_rand48_SoA_size));
	// calculate strided iteration constants
	static const unsigned long long a = 0x5DEECE66DLL, c = 0xB;
	int seed = 123;
	unsigned long long A, C;
	A = 1LL; C = 0LL;
	for (unsigned int i = 0; i &lt; buffer_size_MAX; ++i) {
		C += A*c;
		A *= a;
	}
	h_rand48->A.x = A &amp; 0xFFFFFFLL;
	h_rand48->A.y = (A >> 24) &amp; 0xFFFFFFLL;
	h_rand48->C.x = C &amp; 0xFFFFFFLL;
	h_rand48->C.y = (C >> 24) &amp; 0xFFFFFFLL;
	// prepare first nThreads random numbers from seed
	unsigned long long x = (((unsigned long long)seed) &lt;&lt; 16) | 0x330E;
	for (unsigned int i = 0; i &lt; buffer_size_MAX; ++i) {
		x = a*x + c;
		h_rand48->seeds[i].x = x &amp; 0xFFFFFFLL;
		h_rand48->seeds[i].y = (x >> 24) &amp; 0xFFFFFFLL;
	}
	//copy to device
	gpuErrchk( cudaMemcpy( d_rand48, h_rand48, h_rand48_SoA_size, cudaMemcpyHostToDevice));

    PROFILE_POP_RANGE();

	/* Call all init functions */
	/* Prepare cuda event timers for instrumentation */
#if defined(INSTRUMENT_ITERATIONS) &amp;&amp; INSTRUMENT_ITERATIONS
	cudaEventCreate(&amp;instrument_iteration_start);
	cudaEventCreate(&amp;instrument_iteration_stop);
#endif
#if (defined(INSTRUMENT_AGENT_FUNCTIONS) &amp;&amp; INSTRUMENT_AGENT_FUNCTIONS) || (defined(INSTRUMENT_INIT_FUNCTIONS) &amp;&amp; INSTRUMENT_INIT_FUNCTIONS) || (defined(INSTRUMENT_STEP_FUNCTIONS) &amp;&amp; INSTRUMENT_STEP_FUNCTIONS) || (defined(INSTRUMENT_EXIT_FUNCTIONS) &amp;&amp; INSTRUMENT_EXIT_FUNCTIONS)
	cudaEventCreate(&amp;instrument_start);
	cudaEventCreate(&amp;instrument_stop);
#endif

	<xsl:for-each select="gpu:xmodel/gpu:environment/gpu:initFunctions/gpu:initFunction">
#if defined(INSTRUMENT_INIT_FUNCTIONS) &amp;&amp; INSTRUMENT_INIT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
    PROFILE_PUSH_RANGE("<xsl:value-of select="gpu:name"/>");
    <xsl:value-of select="gpu:name"/>();
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_INIT_FUNCTIONS) &amp;&amp; INSTRUMENT_INIT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&amp;instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: <xsl:value-of select="gpu:name"/> = %f (ms)\n", instrument_milliseconds);
#endif
	</xsl:for-each>

  /* If any Agents can generate IDs, update the device value after init functions have executed */
<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
<xsl:variable name="agent_name" select="xmml:name" />
<xsl:for-each select="xmml:memory/gpu:variable">
<xsl:variable name="variable_name" select="xmml:name" />
<xsl:variable name="variable_type" select="xmml:type" />
<xsl:variable name="type_is_integer"><xsl:call-template name="typeIsInteger"><xsl:with-param name="type" select="$variable_type"/></xsl:call-template></xsl:variable>
<!-- If the agent has a variable name id, of a single integer type -->
<xsl:if test="$variable_name='id' and not(xmml:arrayLength) and $type_is_integer='true'" >
  update_device_generate_<xsl:value-of select="$agent_name"/>_id();
</xsl:if>
</xsl:for-each>
</xsl:for-each>
  
  /* Init CUDA Streams for function layers */
  <xsl:for-each select="gpu:xmodel/xmml:layers/xmml:layer">
  <xsl:sort select="count(gpu:layerFunction)" order="descending"/>
  <xsl:if test="position() =1"> <!-- Get the layer with most functions -->
  <xsl:for-each select="gpu:layerFunction">
  gpuErrchk(cudaStreamCreate(&amp;stream<xsl:value-of select="position()"/>));</xsl:for-each>
  </xsl:if>
  </xsl:for-each>

#if defined(OUTPUT_POPULATION_PER_ITERATION) &amp;&amp; OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state">
		printf("Init agent_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count: %u\n",get_agent_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count());
	</xsl:for-each>
#endif
} 

<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent"><xsl:if test="gpu:type='continuous'"> <xsl:for-each select="xmml:states/gpu:state">
void sort_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count); 
	gridSize = (h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs&lt;&lt;&lt;gridSize, blockSize&gt;&gt;&gt;(d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_keys, d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_values, d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_keys),  thrust::device_pointer_cast(d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_keys) + h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count,  thrust::device_pointer_cast(d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, reorder_<xsl:value-of select="../../xmml:name"/>_agents, no_sm, h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count); 
	gridSize = (h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_<xsl:value-of select="../../xmml:name"/>_agents&lt;&lt;&lt;gridSize, blockSize&gt;&gt;&gt;(d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_values, d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, d_<xsl:value-of select="../../xmml:name"/>s_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* d_<xsl:value-of select="../../xmml:name"/>s_temp = d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>;
	d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/> = d_<xsl:value-of select="../../xmml:name"/>s_swap;
	d_<xsl:value-of select="../../xmml:name"/>s_swap = d_<xsl:value-of select="../../xmml:name"/>s_temp;	
}
</xsl:for-each></xsl:if></xsl:for-each>

void cleanup(){
    PROFILE_SCOPED_RANGE("cleanup");

    /* Call all exit functions */
	<xsl:for-each select="gpu:xmodel/gpu:environment/gpu:exitFunctions/gpu:exitFunction">
#if defined(INSTRUMENT_EXIT_FUNCTIONS) &amp;&amp; INSTRUMENT_EXIT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif

    PROFILE_PUSH_RANGE("<xsl:value-of select="gpu:name"/>");
    <xsl:value-of select="gpu:name"/>();
	PROFILE_POP_RANGE();

#if defined(INSTRUMENT_EXIT_FUNCTIONS) &amp;&amp; INSTRUMENT_EXIT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&amp;instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: <xsl:value-of select="gpu:name"/> = %f (ms)\n", instrument_milliseconds);
#endif
	</xsl:for-each>

	/* Agent data free*/
	<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
	/* <xsl:value-of select="xmml:name"/> Agent variables */
	gpuErrchk(cudaFree(d_<xsl:value-of select="xmml:name"/>s));
	gpuErrchk(cudaFree(d_<xsl:value-of select="xmml:name"/>s_swap));
	gpuErrchk(cudaFree(d_<xsl:value-of select="xmml:name"/>s_new));
	<xsl:for-each select="xmml:states/gpu:state">
	free( h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>);
	gpuErrchk(cudaFree(d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>));
	</xsl:for-each>
	</xsl:for-each>

	/* Message data free */
	<xsl:for-each select="gpu:xmodel/xmml:messages/gpu:message">
	/* <xsl:value-of select="xmml:name"/> Message variables */
	free( h_<xsl:value-of select="xmml:name"/>s);
	gpuErrchk(cudaFree(d_<xsl:value-of select="xmml:name"/>s));
	gpuErrchk(cudaFree(d_<xsl:value-of select="xmml:name"/>s_swap));<xsl:if test="gpu:partitioningSpatial">
	gpuErrchk(cudaFree(d_<xsl:value-of select="xmml:name"/>_partition_matrix));
#ifdef FAST_ATOMIC_SORTING
	gpuErrchk(cudaFree(d_xmachine_message_<xsl:value-of select="xmml:name"/>_local_bin_index));
	gpuErrchk(cudaFree(d_xmachine_message_<xsl:value-of select="xmml:name"/>_unsorted_index));
  gpuErrchk(cudaFree(d_temp_scan_storage_xmachine_message_<xsl:value-of select="xmml:name"/>));
  d_temp_scan_storage_xmachine_message_<xsl:value-of select="xmml:name"/> = nullptr;
  temp_scan_bytes_xmachine_message_<xsl:value-of select="xmml:name"/> = 0;
#else
	gpuErrchk(cudaFree(d_xmachine_message_<xsl:value-of select="xmml:name"/>_keys));
	gpuErrchk(cudaFree(d_xmachine_message_<xsl:value-of select="xmml:name"/>_values));
    gpuErrchk(cudaFree(d_xmachine_message_<xsl:value-of select="xmml:name"/>_keys_swap));
    gpuErrchk(cudaFree(d_xmachine_message_<xsl:value-of select="xmml:name"/>_values_swap));
    gpuErrchk(cudaFree(d_CUB_temp_storage_<xsl:value-of select="xmml:name"/>));
    #endif</xsl:if><xsl:if test="gpu:partitioningGraphEdge">
  gpuErrchk(cudaFree(d_xmachine_message_<xsl:value-of select="xmml:name"/>_bounds));
  gpuErrchk(cudaFree(d_xmachine_message_<xsl:value-of select="xmml:name"/>_scatterer));
  gpuErrchk(cudaFree(d_temp_scan_storage_xmachine_message_<xsl:value-of select="xmml:name"/>));
  d_temp_scan_storage_xmachine_message_<xsl:value-of select="xmml:name"/> = nullptr;
  temp_scan_bytes_xmachine_message_<xsl:value-of select="xmml:name"/> = 0;
  </xsl:if><xsl:text>
	</xsl:text></xsl:for-each>

    /* Free temporary CUB memory if required. */
    <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
    if(d_temp_scan_storage_<xsl:value-of select="xmml:name"/> != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_<xsl:value-of select="xmml:name"/>));
      d_temp_scan_storage_<xsl:value-of select="xmml:name"/> = nullptr;
      temp_scan_storage_bytes_<xsl:value-of select="xmml:name"/> = 0;
    }
    </xsl:for-each>

  /* Graph data free */
  <xsl:for-each select="gpu:xmodel/gpu:environment/gpu:graphs/gpu:staticGraph">
  gpuErrchk(cudaFree(d_staticGraph_memory_<xsl:value-of select="gpu:name"/>));
  d_staticGraph_memory_<xsl:value-of select="gpu:name"/> = nullptr;
  // Free host memory
  free(h_staticGraph_memory_<xsl:value-of select="gpu:name"/>);
  h_staticGraph_memory_<xsl:value-of select="gpu:name"/> = nullptr;
  </xsl:for-each>
  
  /* CUDA Streams for function layers */
  <xsl:for-each select="gpu:xmodel/xmml:layers/xmml:layer">
  <xsl:sort select="count(gpu:layerFunction)" order="descending"/>
  <xsl:if test="position() =1"> <!-- Get the layer with most functions -->
  <xsl:for-each select="gpu:layerFunction">
  gpuErrchk(cudaStreamDestroy(stream<xsl:value-of select="position()"/>));</xsl:for-each>
  </xsl:if>
  </xsl:for-each>

  /* CUDA Event Timers for Instrumentation */
#if defined(INSTRUMENT_ITERATIONS) &amp;&amp; INSTRUMENT_ITERATIONS
	cudaEventDestroy(instrument_iteration_start);
	cudaEventDestroy(instrument_iteration_stop);
#endif
#if (defined(INSTRUMENT_AGENT_FUNCTIONS) &amp;&amp; INSTRUMENT_AGENT_FUNCTIONS) || (defined(INSTRUMENT_INIT_FUNCTIONS) &amp;&amp; INSTRUMENT_INIT_FUNCTIONS) || (defined(INSTRUMENT_STEP_FUNCTIONS) &amp;&amp; INSTRUMENT_STEP_FUNCTIONS) || (defined(INSTRUMENT_EXIT_FUNCTIONS) &amp;&amp; INSTRUMENT_EXIT_FUNCTIONS)
	cudaEventDestroy(instrument_start);
	cudaEventDestroy(instrument_stop);
#endif
}

void singleIteration(){
PROFILE_SCOPED_RANGE("singleIteration");

#if defined(INSTRUMENT_ITERATIONS) &amp;&amp; INSTRUMENT_ITERATIONS
	cudaEventRecord(instrument_iteration_start);
#endif

    // Increment the iteration number.
    g_iterationNumber++;

  /* set all non partitioned, spatial partitioned and On-Graph Partitioned message counts to 0*/<xsl:for-each select="gpu:xmodel/xmml:messages/gpu:message"><xsl:if test="gpu:partitioningNone or gpu:partitioningSpatial or gpu:partitioningGraphEdge">
	h_message_<xsl:value-of select="xmml:name"/>_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_<xsl:value-of select="xmml:name"/>_count, &amp;h_message_<xsl:value-of select="xmml:name"/>_count, sizeof(int)));
	</xsl:if></xsl:for-each>

	/* Call agent functions in order iterating through the layer functions */
	<xsl:for-each select="gpu:xmodel/xmml:layers/xmml:layer">
	/* Layer <xsl:value-of select="position()"/>*/
	<xsl:for-each select="gpu:layerFunction">
#if defined(INSTRUMENT_AGENT_FUNCTIONS) &amp;&amp; INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	<xsl:variable name="function" select="xmml:name"/><xsl:variable name="stream_num" select="position()"/>
	<xsl:choose><xsl:when test="../../../xmml:xagents/gpu:xagent/xmml:functions/gpu:function[xmml:name=$function]"></xsl:when>
	<xsl:otherwise>#error "Layer <xsl:value-of select="position()"/> contains layerFunction '<xsl:value-of select="$function" />' which is not defined by any agent type."
	</xsl:otherwise></xsl:choose>
	<xsl:for-each select="../../../xmml:xagents/gpu:xagent/xmml:functions/gpu:function[xmml:name=$function]">
    PROFILE_PUSH_RANGE("<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>");
	<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>(stream<xsl:value-of select="$stream_num"/>);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) &amp;&amp; INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&amp;instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/> = %f (ms)\n", instrument_milliseconds);
#endif
	</xsl:for-each></xsl:for-each>cudaDeviceSynchronize();
  </xsl:for-each>

  /* If any Agents can generate IDs, update the host value after agent functions have executed */
<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
<xsl:variable name="agent_name" select="xmml:name" />
<xsl:for-each select="xmml:memory/gpu:variable">
<xsl:variable name="variable_name" select="xmml:name" />
<xsl:variable name="variable_type" select="xmml:type" />
<xsl:variable name="type_is_integer"><xsl:call-template name="typeIsInteger"><xsl:with-param name="type" select="$variable_type"/></xsl:call-template></xsl:variable>
<!-- If the agent has a variable name id, of a single integer type -->
<xsl:if test="$variable_name='id' and not(xmml:arrayLength) and $type_is_integer='true'" >
  update_host_generate_<xsl:value-of select="$agent_name"/>_id();
</xsl:if>
</xsl:for-each>
</xsl:for-each>
    
    /* Call all step functions */
	<xsl:for-each select="gpu:xmodel/gpu:environment/gpu:stepFunctions/gpu:stepFunction">
#if defined(INSTRUMENT_STEP_FUNCTIONS) &amp;&amp; INSTRUMENT_STEP_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
    PROFILE_PUSH_RANGE("<xsl:value-of select="gpu:name"/>");
	<xsl:value-of select="gpu:name"/>();<xsl:text>
	</xsl:text>
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_STEP_FUNCTIONS) &amp;&amp; INSTRUMENT_STEP_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&amp;instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: <xsl:value-of select="gpu:name"/> = %f (ms)\n", instrument_milliseconds);
#endif</xsl:for-each>

/* If any Agents can generate IDs, update the device value after step functions have executed */
<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
<xsl:variable name="agent_name" select="xmml:name" />
<xsl:for-each select="xmml:memory/gpu:variable">
<xsl:variable name="variable_name" select="xmml:name" />
<xsl:variable name="variable_type" select="xmml:type" />
<xsl:variable name="type_is_integer"><xsl:call-template name="typeIsInteger"><xsl:with-param name="type" select="$variable_type"/></xsl:call-template></xsl:variable>
<!-- If the agent has a variable name id, of a single integer type -->
<xsl:if test="$variable_name='id' and not(xmml:arrayLength) and $type_is_integer='true'" >
  update_device_generate_<xsl:value-of select="$agent_name"/>_id();
</xsl:if>
</xsl:for-each>
</xsl:for-each>

#if defined(OUTPUT_POPULATION_PER_ITERATION) &amp;&amp; OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state">
		printf("agent_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count: %u\n",get_agent_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count());
	</xsl:for-each>
#endif

#if defined(INSTRUMENT_ITERATIONS) &amp;&amp; INSTRUMENT_ITERATIONS
	cudaEventRecord(instrument_iteration_stop);
	cudaEventSynchronize(instrument_iteration_stop);
	cudaEventElapsedTime(&amp;instrument_iteration_milliseconds, instrument_iteration_start, instrument_iteration_stop);
	printf("Instrumentation: Iteration Time = %f (ms)\n", instrument_iteration_milliseconds);
#endif
}

/* finish whole simulation after this step */
void set_exit_early() {
	g_exit_early = true;
}

bool get_exit_early() {
	return g_exit_early;
}

/* Environment functions */
<!--
<xsl:for-each select="gpu:xmodel/gpu:environment/gpu:constants/gpu:variable">
void set_<xsl:value-of select="xmml:name"/>(<xsl:value-of select="xmml:type"/>* h_<xsl:value-of select="xmml:name"/>){
	gpuErrchk(cudaMemcpyToSymbol(<xsl:value-of select="xmml:name"/>, h_<xsl:value-of select="xmml:name"/>, sizeof(<xsl:value-of select="xmml:type"/>)<xsl:if test="xmml:arrayLength">*<xsl:value-of select="xmml:arrayLength"/></xsl:if>));
}
</xsl:for-each>
-->

<!-- -->
//host constant declaration
<xsl:for-each select="gpu:xmodel/gpu:environment/gpu:constants/gpu:variable">
<xsl:value-of select="xmml:type"/><xsl:text> h_env_</xsl:text><xsl:value-of select="xmml:name"/><xsl:if test="xmml:arrayLength">[<xsl:value-of select="xmml:arrayLength"/>]</xsl:if>;
</xsl:for-each>

<xsl:for-each select="gpu:xmodel/gpu:environment/gpu:constants/gpu:variable">

//constant setter
void set_<xsl:value-of select="xmml:name"/>(<xsl:value-of select="xmml:type"/>* h_<xsl:value-of select="xmml:name"/>){
    gpuErrchk(cudaMemcpyToSymbol(<xsl:value-of select="xmml:name"/>, h_<xsl:value-of select="xmml:name"/>, sizeof(<xsl:value-of select="xmml:type"/>)<xsl:if test="xmml:arrayLength">*<xsl:value-of select="xmml:arrayLength"/></xsl:if>));
    memcpy(&amp;h_env_<xsl:value-of select="xmml:name"/>, h_<xsl:value-of select="xmml:name"/>,sizeof(<xsl:value-of select="xmml:type"/>)<xsl:if test="xmml:arrayLength">*<xsl:value-of select="xmml:arrayLength"/></xsl:if>);
}

//constant getter
const <xsl:value-of select="xmml:type"/>* get_<xsl:value-of select="xmml:name"/>(){
    return <xsl:if test="not(xmml:arrayLength)">&amp;</xsl:if>h_env_<xsl:value-of select="xmml:name"/>;
}

</xsl:for-each>
<!-- -->


/* Agent data access functions*/
<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
    
int get_agent_<xsl:value-of select="xmml:name"/>_MAX_count(){
    return xmachine_memory_<xsl:value-of select="xmml:name"/>_MAX;
}

<xsl:for-each select="xmml:states/gpu:state">
int get_agent_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count(){
	<xsl:if test="../../gpu:type='continuous'">//continuous agent
	return h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count;
	</xsl:if><xsl:if test="../../gpu:type='discrete'">//discrete agent 
	return xmachine_memory_<xsl:value-of select="../../xmml:name"/>_MAX;</xsl:if>
}

xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* get_device_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_agents(){
	return d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>;
}

xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* get_host_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_agents(){
	return h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>;
}
</xsl:for-each>
<xsl:if test="gpu:type='discrete'">
int get_<xsl:value-of select="xmml:name"/>_population_width(){
  return h_xmachine_memory_<xsl:value-of select="xmml:name"/>_pop_width;
}
</xsl:if>

</xsl:for-each>


/* Host based access of agent variables*/
<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent"><xsl:variable name="agent_name" select="xmml:name"/>
<xsl:for-each select="xmml:states/gpu:state"><xsl:variable name="agent_state" select="xmml:name"/>
<xsl:for-each select="../../xmml:memory/gpu:variable"><xsl:variable name="variable_name" select="xmml:name"/><xsl:variable name="variable_type" select="xmml:type" />
<xsl:if test="not(xmml:arrayLength)">
/** <xsl:value-of select="$variable_type"/> get_<xsl:value-of select="$agent_name"/>_<xsl:value-of select="$agent_state"/>_variable_<xsl:value-of select="$variable_name"/>(unsigned int index)
 * Gets the value of the <xsl:value-of select="$variable_name"/> variable of an <xsl:value-of select="$agent_name"/> agent in the <xsl:value-of select="$agent_state"/> state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable <xsl:value-of select="$variable_name"/>
 */
__host__ <xsl:value-of select="$variable_type"/> get_<xsl:value-of select="$agent_name"/>_<xsl:value-of select="$agent_state"/>_variable_<xsl:value-of select="$variable_name"/>(unsigned int index){
    unsigned int count = get_agent_<xsl:value-of select="$agent_name"/>_<xsl:value-of select="$agent_state"/>_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count &gt; 0 &amp;&amp; index &lt; count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$agent_state"/>_variable_<xsl:value-of select="$variable_name"/>_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$agent_state"/>-&gt;<xsl:value-of select="$variable_name"/>,
                    d_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$agent_state"/>-&gt;<xsl:value-of select="$variable_name"/>,
                    count * sizeof(<xsl:value-of select="$variable_type"/>),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$agent_state"/>_variable_<xsl:value-of select="$variable_name"/>_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$agent_state"/>-&gt;<xsl:value-of select="$variable_name"/>[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access <xsl:value-of select="$variable_name"/> for the %u th member of <xsl:value-of select="$agent_name"/>_<xsl:value-of select="$agent_state"/>. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="$variable_type"/></xsl:call-template>;

    }
}
</xsl:if>
<xsl:if test="xmml:arrayLength">
/** <xsl:value-of select="$variable_type"/> get_<xsl:value-of select="$agent_name"/>_<xsl:value-of select="$agent_state"/>_variable_<xsl:value-of select="$variable_name"/>(unsigned int index, unsigned int element)
 * Gets the element-th value of the <xsl:value-of select="$variable_name"/> variable array of an <xsl:value-of select="$agent_name"/> agent in the <xsl:value-of select="$agent_state"/> state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable <xsl:value-of select="$variable_name"/>
 */
__host__ <xsl:value-of select="$variable_type"/> get_<xsl:value-of select="$agent_name"/>_<xsl:value-of select="$agent_state"/>_variable_<xsl:value-of select="$variable_name"/>(unsigned int index, unsigned int element){
    unsigned int count = get_agent_<xsl:value-of select="$agent_name"/>_<xsl:value-of select="$agent_state"/>_count();
    unsigned int numElements = <xsl:value-of select="xmml:arrayLength"/>;
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count &gt; 0 &amp;&amp; index &lt; count &amp;&amp; element &lt; numElements ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$agent_state"/>_variable_<xsl:value-of select="$variable_name"/>_data_iteration != currentIteration){
            <!-- @optimisation - If the count is close enough to MAX, it would be better to issue a single large memcpy. -->
            for(unsigned int e = 0; e &lt; numElements; e++){
                gpuErrchk(
                    cudaMemcpy(
                        h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$agent_state"/>-&gt;<xsl:value-of select="$variable_name"/> + (e * xmachine_memory_<xsl:value-of select="$agent_name"/>_MAX),
                        d_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$agent_state"/>-&gt;<xsl:value-of select="$variable_name"/> + (e * xmachine_memory_<xsl:value-of select="$agent_name"/>_MAX), 
                        count * sizeof(<xsl:value-of select="$variable_type"/>), 
                        cudaMemcpyDeviceToHost
                    )
                );
                // Update some global value indicating what data is currently present in that host array.
                h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$agent_state"/>_variable_<xsl:value-of select="$variable_name"/>_data_iteration = currentIteration;
            }
        }

        // Return the value of the index-th element of the relevant host array.
        return h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$agent_state"/>-&gt;<xsl:value-of select="$variable_name"/>[index + (element * xmachine_memory_<xsl:value-of select="$agent_name"/>_MAX)];

    } else {
        fprintf(stderr, "Warning: Attempting to access the %u-th element of <xsl:value-of select="$variable_name"/> for the %u th member of <xsl:value-of select="$agent_name"/>_<xsl:value-of select="$agent_state"/>. count is %u at iteration %u\n", element, index, count, currentIteration);
        // Otherwise we return a default value
        return <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="$variable_type"/></xsl:call-template>;

    }
}
</xsl:if>
</xsl:for-each>
</xsl:for-each>
</xsl:for-each>


/* Host based agent creation functions */
// These are only available for continuous agents.

<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
<xsl:if test="gpu:type='continuous'">

/* copy_single_xmachine_memory_<xsl:value-of select="xmml:name"/>_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_<xsl:value-of select="xmml:name"/>_hostToDevice(xmachine_memory_<xsl:value-of select="xmml:name"/>_list * d_dst, xmachine_memory_<xsl:value-of select="xmml:name"/> * h_agent){
<xsl:for-each select="xmml:memory/gpu:variable"><xsl:if test="xmml:arrayLength"> 
	for(unsigned int i = 0; i &lt; <xsl:value-of select="xmml:arrayLength"/>; i++){
		gpuErrchk(cudaMemcpy(d_dst-&gt;<xsl:value-of select="xmml:name"/> + (i * xmachine_memory_<xsl:value-of select="../../xmml:name" />_MAX), h_agent-&gt;<xsl:value-of select="xmml:name"/> + i, sizeof(<xsl:value-of select="xmml:type"/>), cudaMemcpyHostToDevice));
    }
</xsl:if><xsl:if test="not(xmml:arrayLength)"> 
		gpuErrchk(cudaMemcpy(d_dst-&gt;<xsl:value-of select="xmml:name"/>, &amp;h_agent-&gt;<xsl:value-of select="xmml:name"/>, sizeof(<xsl:value-of select="xmml:type"/>), cudaMemcpyHostToDevice));
</xsl:if>
</xsl:for-each>
}
/*
 * Private function to copy some elements from a host based struct of arrays to a device based struct of arrays for a single agent state.
 * Individual copies of `count` elements are performed for each agent variable or each component of agent array variables, to avoid wasted data transfer.
 * There will be a point at which a single cudaMemcpy will outperform many smaller memcpys, however host based agent creation should typically only populate a fraction of the maximum buffer size, so this should be more efficient.
 * @optimisation - experimentally find the proportion at which transferring the whole SoA would be better and incorporate this. The same will apply to agent variable arrays.
 * 
 * @param d_dst device destination SoA
 * @oaram h_src host source SoA
 * @param count the number of agents to transfer data for
 */
void copy_partial_xmachine_memory_<xsl:value-of select="xmml:name"/>_hostToDevice(xmachine_memory_<xsl:value-of select="xmml:name"/>_list * d_dst, xmachine_memory_<xsl:value-of select="xmml:name"/>_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count &gt; 0){
	<xsl:for-each select="xmml:memory/gpu:variable"><xsl:if test="xmml:arrayLength"> 
		for(unsigned int i = 0; i &lt; <xsl:value-of select="xmml:arrayLength"/>; i++){
			gpuErrchk(cudaMemcpy(d_dst-&gt;<xsl:value-of select="xmml:name"/> + (i * xmachine_memory_<xsl:value-of select="../../xmml:name" />_MAX), h_src-&gt;<xsl:value-of select="xmml:name"/> + (i * xmachine_memory_<xsl:value-of select="../../xmml:name" />_MAX), count * sizeof(<xsl:value-of select="xmml:type"/>), cudaMemcpyHostToDevice));
        }

</xsl:if><xsl:if test="not(xmml:arrayLength)"> 
		gpuErrchk(cudaMemcpy(d_dst-&gt;<xsl:value-of select="xmml:name"/>, h_src-&gt;<xsl:value-of select="xmml:name"/>, count * sizeof(<xsl:value-of select="xmml:type"/>), cudaMemcpyHostToDevice));
</xsl:if>
	</xsl:for-each>
    }
}
</xsl:if>
</xsl:for-each>

<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent"><xsl:variable name="agent_name" select="xmml:name"/>
<xsl:if test="gpu:type='continuous'">
xmachine_memory_<xsl:value-of select="$agent_name" />* h_allocate_agent_<xsl:value-of select="$agent_name" />(){
	xmachine_memory_<xsl:value-of select="$agent_name" />* agent = (xmachine_memory_<xsl:value-of select="$agent_name" />*)malloc(sizeof(xmachine_memory_<xsl:value-of select="$agent_name" />));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_<xsl:value-of select="$agent_name" />));
<xsl:for-each select="xmml:memory/gpu:variable">
<xsl:if test="xmml:defaultValue and not(xmml:arrayLength)">
    agent-&gt;<xsl:value-of select="xmml:name"/> = <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/><xsl:with-param name="defaultValue" select="xmml:defaultValue" /></xsl:call-template>;
</xsl:if>
<xsl:if test="xmml:arrayLength">	// Agent variable arrays must be allocated
    agent-&gt;<xsl:value-of select="xmml:name"/> = (<xsl:value-of select="xmml:type"/>*)malloc(<xsl:value-of select="xmml:arrayLength"/> * sizeof(<xsl:value-of select="xmml:type"/>));
	<xsl:choose><xsl:when test="xmml:defaultValue">// If we have a default value, set each element correctly.
	for(unsigned int index = 0; index &lt; <xsl:value-of select="xmml:arrayLength"/>; index++){
		agent-&gt;<xsl:value-of select="xmml:name"/>[index] = <xsl:call-template name="defaultInitialiser"><xsl:with-param name="type" select="xmml:type"/><xsl:with-param name="defaultValue" select="xmml:defaultValue" /></xsl:call-template>;
	}</xsl:when><xsl:otherwise>
    // If there is no default value, memset to 0.
    memset(agent-&gt;<xsl:value-of select="xmml:name"/>, 0, sizeof(<xsl:value-of select="xmml:type"/>)*<xsl:value-of select="xmml:arrayLength"/>);</xsl:otherwise>
	</xsl:choose>
</xsl:if>
</xsl:for-each>
	return agent;
}
void h_free_agent_<xsl:value-of select="$agent_name" />(xmachine_memory_<xsl:value-of select="$agent_name" />** agent){
<xsl:variable name="xagentname" select="xmml:xagentName"/><xsl:for-each select="xmml:memory/gpu:variable"><xsl:if test="xmml:arrayLength">
    free((*agent)-&gt;<xsl:value-of select="xmml:name"/>);
</xsl:if></xsl:for-each> 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_<xsl:value-of select="$agent_name" />** h_allocate_agent_<xsl:value-of select="$agent_name" />_array(unsigned int count){
	xmachine_memory_<xsl:value-of select="$agent_name" /> ** agents = (xmachine_memory_<xsl:value-of select="$agent_name" />**)malloc(count * sizeof(xmachine_memory_<xsl:value-of select="$agent_name" />*));
	for (unsigned int i = 0; i &lt; count; i++) {
		agents[i] = h_allocate_agent_<xsl:value-of select="$agent_name" />();
	}
	return agents;
}
void h_free_agent_<xsl:value-of select="$agent_name" />_array(xmachine_memory_<xsl:value-of select="$agent_name" />*** agents, unsigned int count){
	for (unsigned int i = 0; i &lt; count; i++) {
		h_free_agent_<xsl:value-of select="$agent_name" />(&amp;((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_<xsl:value-of select="$agent_name"/>_AoS_to_SoA(xmachine_memory_<xsl:value-of select="$agent_name"/>_list * dst, xmachine_memory_<xsl:value-of select="$agent_name" />** src, unsigned int count){
	if(count &gt; 0){
		for(unsigned int i = 0; i &lt; count; i++){
			<xsl:for-each select="xmml:memory/gpu:variable"><xsl:if test="xmml:arrayLength"> 
			for(unsigned int j = 0; j &lt; <xsl:value-of select="xmml:arrayLength" />; j++){
				dst-&gt;<xsl:value-of select="xmml:name"/>[(j * xmachine_memory_<xsl:value-of select="../../xmml:name" />_MAX) + i] = src[i]-&gt;<xsl:value-of select="xmml:name"/>[j];
			}
			</xsl:if><xsl:if test="not(xmml:arrayLength)"> 
			dst-&gt;<xsl:value-of select="xmml:name"/>[i] = src[i]-&gt;<xsl:value-of select="xmml:name"/>;
			</xsl:if>
			</xsl:for-each>
		}
	}
}
<xsl:for-each select="xmml:states/gpu:state"><xsl:variable name="state" select="xmml:name"/>

void h_add_agent_<xsl:value-of select="$agent_name" />_<xsl:value-of select="$state" />(xmachine_memory_<xsl:value-of select="$agent_name" />* agent){
	if (h_xmachine_memory_<xsl:value-of select="$agent_name"/>_count + 1 &gt; xmachine_memory_<xsl:value-of select="$agent_name"/>_MAX){
		printf("Error: Buffer size of <xsl:value-of select="$agent_name"/> agents in state <xsl:value-of select="$state"/> will be exceeded by h_add_agent_<xsl:value-of select="$agent_name" />_<xsl:value-of select="$state" />\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_<xsl:value-of select="$agent_name"/>_hostToDevice(d_<xsl:value-of select="$agent_name"/>s_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&amp;minGridSize, &amp;blockSize, append_<xsl:value-of select="$agent_name"/>_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_<xsl:value-of select="$agent_name"/>_Agents &lt;&lt;&lt;gridSize, blockSize, 0, stream1 &gt;&gt;&gt;(d_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state"/>, d_<xsl:value-of select="$agent_name"/>s_new, h_xmachine_memory_<xsl:value-of select="$agent_name"/>_<xsl:value-of select="$state"/>_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_<xsl:value-of select="$agent_name"/>_<xsl:value-of select="$state"/>_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_<xsl:value-of select="$agent_name"/>_<xsl:value-of select="$state"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="$agent_name"/>_<xsl:value-of select="$state"/>_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    <xsl:for-each select="../../xmml:memory/gpu:variable"><xsl:variable name="variable_name" select="xmml:name"/><xsl:variable name="variable_type" select="xmml:type" />h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state"/>_variable_<xsl:value-of select="$variable_name"/>_data_iteration = 0;
    </xsl:for-each>

}
void h_add_agents_<xsl:value-of select="$agent_name" />_<xsl:value-of select="$state" />(xmachine_memory_<xsl:value-of select="$agent_name" />** agents, unsigned int count){
	if(count &gt; 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_<xsl:value-of select="$agent_name"/>_count + count &gt; xmachine_memory_<xsl:value-of select="$agent_name"/>_MAX){
			printf("Error: Buffer size of <xsl:value-of select="$agent_name"/> agents in state <xsl:value-of select="$state"/> will be exceeded by h_add_agents_<xsl:value-of select="$agent_name" />_<xsl:value-of select="$state" />\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_<xsl:value-of select="$agent_name"/>_AoS_to_SoA(h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state"/>, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_<xsl:value-of select="$agent_name"/>_hostToDevice(d_<xsl:value-of select="$agent_name"/>s_new, h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state"/>, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&amp;minGridSize, &amp;blockSize, append_<xsl:value-of select="$agent_name"/>_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_<xsl:value-of select="$agent_name"/>_Agents &lt;&lt;&lt;gridSize, blockSize, 0, stream1 &gt;&gt;&gt;(d_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state"/>, d_<xsl:value-of select="$agent_name"/>s_new, h_xmachine_memory_<xsl:value-of select="$agent_name"/>_<xsl:value-of select="$state"/>_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_<xsl:value-of select="$agent_name"/>_<xsl:value-of select="$state"/>_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_<xsl:value-of select="$agent_name"/>_<xsl:value-of select="$state"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="$agent_name"/>_<xsl:value-of select="$state"/>_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        <xsl:for-each select="../../xmml:memory/gpu:variable"><xsl:variable name="variable_name" select="xmml:name"/><xsl:variable name="variable_type" select="xmml:type" />h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state"/>_variable_<xsl:value-of select="$variable_name"/>_data_iteration = 0;
        </xsl:for-each>

	}
}
</xsl:for-each>
</xsl:if>
</xsl:for-each>

/*  Analytics Functions */

<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
  <xsl:variable name="agent_name" select="xmml:name"/>
<xsl:for-each select="xmml:states/gpu:state">
  <xsl:variable name="state" select="xmml:name"/>
<xsl:for-each select="../../xmml:memory/gpu:variable">
<xsl:if test="not(xmml:arrayLength)"> <!-- Disable agent array reductions -->
<xsl:value-of select="xmml:type"/> reduce_<xsl:value-of select="$agent_name"/>_<xsl:value-of select="$state"/>_<xsl:value-of select="xmml:name"/>_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state"/>-><xsl:value-of select="xmml:name"/>),  thrust::device_pointer_cast(d_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state"/>-><xsl:value-of select="xmml:name"/>) + h_xmachine_memory_<xsl:value-of select="$agent_name"/>_<xsl:value-of select="$state"/>_count);
}

<xsl:if test="contains(xmml:type, 'int')">
<xsl:value-of select="xmml:type"/> count_<xsl:value-of select="$agent_name"/>_<xsl:value-of select="$state"/>_<xsl:value-of select="xmml:name"/>_variable(<xsl:value-of select="xmml:type"/> count_value){
    //count in default stream
    return (<xsl:value-of select="xmml:type"/>)thrust::count(thrust::device_pointer_cast(d_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state"/>-><xsl:value-of select="xmml:name"/>),  thrust::device_pointer_cast(d_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state"/>-><xsl:value-of select="xmml:name"/>) + h_xmachine_memory_<xsl:value-of select="$agent_name"/>_<xsl:value-of select="$state"/>_count, count_value);
}
</xsl:if>

<xsl:if test="not(contains(xmml:type, 'vec'))"> <!-- Any non-vector data type can be min/maxed. -->
<xsl:value-of select="xmml:type"/> min_<xsl:value-of select="$agent_name"/>_<xsl:value-of select="$state"/>_<xsl:value-of select="xmml:name"/>_variable(){
    //min in default stream
    thrust::device_ptr&lt;<xsl:value-of select="xmml:type"/>&gt; thrust_ptr = thrust::device_pointer_cast(d_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state"/>-&gt;<xsl:value-of select="xmml:name"/>);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_<xsl:value-of select="$agent_name"/>_<xsl:value-of select="$state"/>_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
<xsl:value-of select="xmml:type"/> max_<xsl:value-of select="$agent_name"/>_<xsl:value-of select="$state"/>_<xsl:value-of select="xmml:name"/>_variable(){
    //max in default stream
    thrust::device_ptr&lt;<xsl:value-of select="xmml:type"/>&gt; thrust_ptr = thrust::device_pointer_cast(d_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state"/>-&gt;<xsl:value-of select="xmml:name"/>);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_<xsl:value-of select="$agent_name"/>_<xsl:value-of select="$state"/>_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
</xsl:if>


</xsl:if>

</xsl:for-each>
</xsl:for-each>
</xsl:for-each>


/* Agent functions */

<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:functions/gpu:function">
	
/* Shared memory size calculator for agent function */
int <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  <xsl:if test="xmml:inputs/gpu:input"><xsl:variable name="messageName" select="xmml:inputs/gpu:input/xmml:messageName"/>
	<xsl:if test="../../gpu:type='continuous'"><xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messageName]">
	<xsl:if test="gpu:partitioningNone">//Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>));
	</xsl:if><xsl:if test="gpu:partitioningDiscrete">//Continuous agent and message input has discrete partitioning
	//Will be reading using texture lookups so sm size can stay the same but need to hold range and width
	sm_size += (blockSize * sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>));
	</xsl:if><xsl:if test="gpu:partitioningSpatial">//Continuous agent and message input is spatially partitioned
	sm_size += (blockSize * sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>));
	</xsl:if>
  <xsl:if test="gpu:partitioningGraphEdge">//Continuous agent and message input is On-Graph Partitioned
  sm_size += (blockSize * sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>));
  </xsl:if>
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	</xsl:for-each>
	</xsl:if><xsl:if test="../../gpu:type='discrete'">
	<xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messageName]">
  <xsl:if test="gpu:partitioningNone  or gpu:partitioningSpatial or gpu:partitioningGraphEdge">//Discrete agent continuous message input
	sm_size += (blockSize * sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>));
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	</xsl:if><xsl:if test="gpu:partitioningDiscrete">//Discrete agent and message input has discrete partitioning
	int sm_grid_width = (int)ceil(sqrt(blockSize));
	int sm_grid_size = (int)pow((float)sm_grid_width+(h_message_<xsl:value-of select="xmml:name"/>_range*2), 2);
	sm_size += (sm_grid_size *sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>)); //update sm size
	sm_size += (sm_grid_size * PADDING);  //offset for avoiding conflicts
	</xsl:if></xsl:for-each></xsl:if></xsl:if>
	return sm_size;
}

/** <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>
 * Agent function prototype for <xsl:value-of select="xmml:name"/> function of <xsl:value-of select="../../xmml:name"/> agent
 */
void <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>(cudaStream_t &amp;stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	<!-- This check has been removed so that we do net get unspecified launch failures when a population of 0 discrete agents is used.
	Alternatively this should be expanded to elseif, with an error message and a graceful exit of the simulator.
	<xsl:if test="../../gpu:type='continuous'"> -->
	if (h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count == 0)
	{
		return;
	}
	<!-- </xsl:if> -->
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count;

	<xsl:if test="xmml:xagentOutputs/gpu:xagentOutput">
	<xsl:for-each select="xmml:xagentOutputs/gpu:xagentOutput">
	<xsl:variable name="xagent_output" select="xmml:xagentName"/><xsl:if test="../../../../../gpu:xagent[xmml:name=$xagent_output]/gpu:type='continuous'">
	//FOR <xsl:value-of select="xmml:xagentName"/> AGENT OUTPUT, RESET THE AGENT NEW LIST SCAN INPUT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, reset_<xsl:value-of select="xmml:xagentName"/>_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_<xsl:value-of select="xmml:xagentName"/>_scan_input&lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_<xsl:value-of select="xmml:xagentName"/>s_new);
	gpuErrchkLaunch();
	</xsl:if></xsl:for-each></xsl:if>

	//******************************** AGENT FUNCTION CONDITION *********************
	<xsl:choose>
	<xsl:when test="xmml:condition"><xsl:if test="../../gpu:type='continuous'">//CONTINUOUS AGENT FUNCTION AND THERE IS A FUNCTION CONDITION
  	
	//COPY CURRENT STATE COUNT TO WORKING COUNT (host and device)
	h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count = h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, sizeof(int)));	
	
	//RESET SCAN INPUTS
	//reset scan input for currentState
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, reset_<xsl:value-of select="../../xmml:name"/>_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_<xsl:value-of select="../../xmml:name"/>_scan_input&lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>);
	gpuErrchkLaunch();
	//reset scan input for working lists
	reset_<xsl:value-of select="../../xmml:name"/>_scan_input&lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s);
	gpuErrchkLaunch();

	//APPLY FUNCTION FILTER
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, <xsl:value-of select="xmml:name"/>_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	<xsl:value-of select="xmml:name"/>_function_filter&lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>, d_<xsl:value-of select="../../xmml:name"/>s);
	gpuErrchkLaunch();

	//GRID AND BLOCK SIZE FOR COMPACT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, scatter_<xsl:value-of select="../../xmml:name"/>_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	
	//COMPACT CURRENT STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_<xsl:value-of select="../../xmml:name"/>, 
        temp_scan_storage_bytes_<xsl:value-of select="../../xmml:name"/>, 
        d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>->_scan_input,
        d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>->_position,
        h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &amp;scan_last_sum, &amp;d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>->_position[h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &amp;scan_last_included, &amp;d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>->_scan_input[h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count = scan_last_sum+1;
	else		
		h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count = scan_last_sum;
	//Scatter into swap
	scatter_<xsl:value-of select="../../xmml:name"/>_Agents&lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s_swap, d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>, 0, h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count);
	gpuErrchkLaunch();
	//use a temp pointer change working swap list with current state list
	xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* <xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>_temp = d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>;
	d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/> = d_<xsl:value-of select="../../xmml:name"/>s_swap;
	d_<xsl:value-of select="../../xmml:name"/>s_swap = <xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count, sizeof(int)));	
		
	//COMPACT WORKING STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_<xsl:value-of select="../../xmml:name"/>, 
        temp_scan_storage_bytes_<xsl:value-of select="../../xmml:name"/>, 
        d_<xsl:value-of select="../../xmml:name"/>s->_scan_input,
        d_<xsl:value-of select="../../xmml:name"/>s->_position,
        h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &amp;scan_last_sum, &amp;d_<xsl:value-of select="../../xmml:name"/>s->_position[h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &amp;scan_last_included, &amp;d_<xsl:value-of select="../../xmml:name"/>s->_scan_input[h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//Scatter into swap
	scatter_<xsl:value-of select="../../xmml:name"/>_Agents&lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s_swap, d_<xsl:value-of select="../../xmml:name"/>s, 0, h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count);
	gpuErrchkLaunch();
	//update working agent count after the scatter
	if (scan_last_included == 1)
		h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count = scan_last_sum+1;
	else		
		h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count = scan_last_sum;
    //use a temp pointer change working swap list with current state list
	xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* <xsl:value-of select="../../xmml:name"/>s_temp = d_<xsl:value-of select="../../xmml:name"/>s;
	d_<xsl:value-of select="../../xmml:name"/>s = d_<xsl:value-of select="../../xmml:name"/>s_swap;
	d_<xsl:value-of select="../../xmml:name"/>s_swap = <xsl:value-of select="../../xmml:name"/>s_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, sizeof(int)));	
	
	//CHECK WORKING LIST COUNT IS NOT EQUAL TO 0
	if (h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count == 0)
	{
		return;
	}
	
	<xsl:if test="../../gpu:type='continuous'">//Update the state list size for occupancy calculations
	state_list_size = h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count;
	</xsl:if>
			
	</xsl:if></xsl:when><xsl:when test="gpu:globalCondition">//THERE IS A GLOBAL CONDITION
	
	//COPY CURRENT STATE COUNT TO WORKING COUNT (host and device)
	h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count = h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, sizeof(int)));	
	
	//RESET SCAN INPUTS
	//reset scan input for currentState
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, reset_<xsl:value-of select="../../xmml:name"/>_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_<xsl:value-of select="../../xmml:name"/>_scan_input&lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>);
	gpuErrchkLaunch();
	
	//APPLY FUNCTION FILTER
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, <xsl:value-of select="xmml:name"/>_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	<xsl:value-of select="xmml:name"/>_function_filter&lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>);
	gpuErrchkLaunch();
	
	//GET CONDTIONS TRUE COUNT FROM CURRENT STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_<xsl:value-of select="../../xmml:name"/>, 
        temp_scan_storage_bytes_<xsl:value-of select="../../xmml:name"/>, 
        d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>->_scan_input,
        d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>->_position,
        h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &amp;scan_last_sum, &amp;d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>->_position[h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &amp;scan_last_included, &amp;d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>->_scan_input[h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	int global_conditions_true = 0;
	if (scan_last_included == 1)
		global_conditions_true = scan_last_sum+1;
	else		
		global_conditions_true = scan_last_sum;
	//check if condition is true for all agents or if max condition count is reached
	if ((global_conditions_true <xsl:choose><xsl:when test="gpu:globalCondition/gpu:mustEvaluateTo='true'">!</xsl:when><xsl:otherwise>=</xsl:otherwise></xsl:choose>= h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count)&amp;&amp;(h_<xsl:value-of select="xmml:name"/>_condition_count &lt; <xsl:value-of select="gpu:globalCondition/gpu:maxItterations"/>))
	{
		h_<xsl:value-of select="xmml:name"/>_condition_count ++;
		return;
	}
	if ((h_<xsl:value-of select="xmml:name"/>_condition_count == <xsl:value-of select="gpu:globalCondition/gpu:maxItterations"/>))
	{
		printf("Global agent condition for <xsl:value-of select="xmml:name"/> function reached the maximum number of <xsl:value-of select="gpu:globalCondition/gpu:maxItterations"/> conditions\n");
	}
	
	//RESET THE CONDITION COUNT
	h_<xsl:value-of select="xmml:name"/>_condition_count = 0;
	
	//MAP CURRENT STATE TO WORKING LIST
	xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* <xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>_temp = d_<xsl:value-of select="../../xmml:name"/>s;
	d_<xsl:value-of select="../../xmml:name"/>s = d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>;
	d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/> = <xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>_temp;
	//set current state count to 0
	h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, sizeof(int)));	
	
	
	</xsl:when><xsl:otherwise>//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* <xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>_temp = d_<xsl:value-of select="../../xmml:name"/>s;
	d_<xsl:value-of select="../../xmml:name"/>s = d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>;
	d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/> = <xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>_temp;
	//set working count to current state count
	h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count = h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count, sizeof(int)));	
	</xsl:otherwise>
	</xsl:choose>
 

	//******************************** AGENT FUNCTION *******************************

	<xsl:if test="xmml:outputs/gpu:output"><xsl:if test="../../gpu:type='continuous'">
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_<xsl:value-of select="xmml:outputs/gpu:output/xmml:messageName"/>_count + h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count > xmachine_message_<xsl:value-of select="xmml:outputs/gpu:output/xmml:messageName"/>_MAX){
		printf("Error: Buffer size of <xsl:value-of select="xmml:outputs/gpu:output/xmml:messageName"/> message will be exceeded in function <xsl:value-of select="xmml:name"/>\n");
		exit(EXIT_FAILURE);
	}
	</xsl:if></xsl:if>
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, GPUFLAME_<xsl:value-of select="xmml:name"/>, <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_sm_size, state_list_size);<xsl:if test="../../gpu:type='continuous'">
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	</xsl:if><xsl:if test="../../gpu:type='discrete'">
	blockSize = lowest_sqr_pow2(blockSize); //For discrete agents the block size must be a square power of 2
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = (int)sqrt(blockSize);
	b.y = b.x;
	g.x = (int)sqrt(gridSize);
	g.y = g.x;</xsl:if>
	sm_size = <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_sm_size(blockSize);
	
	
	<xsl:if test="xmml:inputs/gpu:input"><xsl:variable name="messageName" select="xmml:inputs/gpu:input/xmml:messageName"/>
	<xsl:if test="../../gpu:type='discrete'"><xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messageName]"><xsl:if test="gpu:partitioningDiscrete">
	//check that the range is not greater than the square of the block size. If so then there will be too many uncoalesded reads
	if (h_message_<xsl:value-of select="xmml:name"/>_range > (int)blockSize){
		printf("ERROR: Message range is greater than the thread block size. Increase thread block size or reduce the range!\n");
		exit(EXIT_FAILURE);
	}
	</xsl:if></xsl:for-each></xsl:if></xsl:if>
	
	<xsl:if test="xmml:inputs/gpu:input"><xsl:variable name="messageName" select="xmml:inputs/gpu:input/xmml:messageName"/>
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	<xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messageName]">
	<xsl:if test="gpu:partitioningDiscrete or gpu:partitioningSpatial">//any agent with discrete or partitioned message input uses texture caching
	<xsl:for-each select="xmml:variables/gpu:variable">size_t tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_byte_offset;    
	gpuErrchk( cudaBindTexture(&amp;tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_byte_offset, tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>, d_<xsl:value-of select="../../xmml:name"/>s-><xsl:value-of select="xmml:name"/>, sizeof(<xsl:value-of select="xmml:type"/>)*xmachine_message_<xsl:value-of select="../../xmml:name"/>_MAX));
	h_tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_offset = (int)tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_byte_offset / sizeof(<xsl:value-of select="xmml:type"/>);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_offset, &amp;h_tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_offset, sizeof(int)));
	</xsl:for-each><xsl:if test="gpu:partitioningSpatial">//bind pbm start and end indices to textures
	size_t tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_start_byte_offset;
	size_t tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_end_or_count_byte_offset;
	gpuErrchk( cudaBindTexture(&amp;tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_start_byte_offset, tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_start, d_<xsl:value-of select="xmml:name"/>_partition_matrix->start, sizeof(int)*xmachine_message_<xsl:value-of select="xmml:name"/>_grid_size));
	h_tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_start_offset = (int)tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_start_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_start_offset, &amp;h_tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_start_offset, sizeof(int)));
	gpuErrchk( cudaBindTexture(&amp;tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_end_or_count_byte_offset, tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_end_or_count, d_<xsl:value-of select="xmml:name"/>_partition_matrix->end_or_count, sizeof(int)*xmachine_message_<xsl:value-of select="xmml:name"/>_grid_size));
  h_tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_end_or_count_offset = (int)tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_end_or_count_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_end_or_count_offset, &amp;h_tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_end_or_count_offset, sizeof(int)));

	</xsl:if></xsl:if>
	</xsl:for-each></xsl:if>
	
	<xsl:if test="xmml:outputs/gpu:output"><xsl:variable name="messageName" select="xmml:outputs/gpu:output/xmml:messageName"/><xsl:variable name="outputType" select="xmml:outputs/gpu:output/gpu:type"/>
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	<xsl:if test="../../gpu:type='continuous'"><xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messageName]">
  <xsl:if test="gpu:partitioningNone or gpu:partitioningSpatial or gpu:partitioningGraphEdge">//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_<xsl:value-of select="xmml:name"/>_output_type = <xsl:value-of select="$outputType"/>;
	gpuErrchk( cudaMemcpyToSymbol( d_message_<xsl:value-of select="xmml:name"/>_output_type, &amp;h_message_<xsl:value-of select="xmml:name"/>_output_type, sizeof(int)));
	<xsl:if test="$outputType='optional_message'">//message is optional so reset the swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, reset_<xsl:value-of select="xmml:name"/>_swaps, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_<xsl:value-of select="xmml:name"/>_swaps&lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_<xsl:value-of select="xmml:name"/>s); <!-- Twin Karmakharm Change - Bug found, need to reset the actual message array and not the swap array -->
	gpuErrchkLaunch();
	</xsl:if></xsl:if></xsl:for-each>
	</xsl:if></xsl:if>
	
	
	<xsl:if test="../../gpu:type='continuous'"><xsl:if test="gpu:reallocate='true'">
	//IF CONTINUOUS AGENT CAN REALLOCATE (process dead agents) THEN RESET AGENT SWAPS	
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, reset_<xsl:value-of select="../../xmml:name"/>_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_<xsl:value-of select="../../xmml:name"/>_scan_input&lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s);
	gpuErrchkLaunch();
	</xsl:if></xsl:if>
	
	//MAIN XMACHINE FUNCTION CALL (<xsl:value-of select="xmml:name"/>)
	//Reallocate   : <xsl:choose><xsl:when test="gpu:reallocate='true'">true</xsl:when><xsl:otherwise>false</xsl:otherwise></xsl:choose>
	//Input        : <xsl:value-of select="xmml:inputs/gpu:input/xmml:messageName"/>
	//Output       : <xsl:value-of select="xmml:outputs/gpu:output/xmml:messageName"/>
	//Agent Output : <xsl:value-of select="xmml:xagentOutputs/gpu:xagentOutput/xmml:xagentName"/>
	GPUFLAME_<xsl:value-of select="xmml:name"/>&lt;&lt;&lt;g, b, sm_size, stream&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s<xsl:if test="xmml:xagentOutputs/gpu:xagentOutput">, d_<xsl:value-of select="xmml:xagentOutputs/gpu:xagentOutput/xmml:xagentName"/>s_new</xsl:if>
		<xsl:if test="xmml:inputs/gpu:input"><xsl:variable name="messagename" select="xmml:inputs/gpu:input/xmml:messageName"/>, d_<xsl:value-of select="xmml:inputs/gpu:input/xmml:messageName"/>s<xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messagename]"><xsl:if test="gpu:partitioningSpatial">, d_<xsl:value-of select="xmml:name"/>_partition_matrix</xsl:if><xsl:if test="gpu:partitioningGraphEdge">, d_xmachine_message_<xsl:value-of select="xmml:name"/>_bounds</xsl:if></xsl:for-each></xsl:if>
		<xsl:if test="xmml:outputs/gpu:output">, d_<xsl:value-of select="xmml:outputs/gpu:output/xmml:messageName"/>s<xsl:if test="xmml:outputs/gpu:output/xmml:type='optional_message'">_swap</xsl:if></xsl:if>
		<xsl:if test="gpu:RNG='true'">, d_rand48</xsl:if>);
	gpuErrchkLaunch();
	
	<xsl:if test="xmml:inputs/gpu:input"><xsl:variable name="messageName" select="xmml:inputs/gpu:input/xmml:messageName"/>
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	<xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messageName]">
	<xsl:if test="gpu:partitioningDiscrete or gpu:partitioningSpatial">//any agent with discrete or partitioned message input uses texture caching
	<xsl:for-each select="xmml:variables/gpu:variable">gpuErrchk( cudaUnbindTexture(tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>));
	</xsl:for-each><xsl:if test="gpu:partitioningSpatial">//unbind pbm indices
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_start));
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_end_or_count));
    </xsl:if></xsl:if>
	</xsl:for-each></xsl:if>

	<xsl:if test="xmml:outputs/gpu:output"><xsl:variable name="messageName" select="xmml:outputs/gpu:output/xmml:messageName"/><xsl:variable name="outputType" select="xmml:outputs/gpu:output/gpu:type"/><xsl:variable name="xagentName" select="../../xmml:name"/>
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	<xsl:if test="../../gpu:type='continuous'"><xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messageName]">
  <xsl:if test="gpu:partitioningNone or gpu:partitioningSpatial or gpu:partitioningGraphEdge">
	<xsl:if test="$outputType='optional_message'">//<xsl:value-of select="xmml:name"/> Message Type Prefix Sum
	<!-- Twin Karmakharm bug fix 16/09/2014 - Bug found need to swap the message array so that it gets scanned properly -->
	//swap output
	xmachine_message_<xsl:value-of select="xmml:name"/>_list* d_<xsl:value-of select="xmml:name"/>s_scanswap_temp = d_<xsl:value-of select="xmml:name"/>s;
	d_<xsl:value-of select="xmml:name"/>s = d_<xsl:value-of select="xmml:name"/>s_swap;
	d_<xsl:value-of select="xmml:name"/>s_swap = d_<xsl:value-of select="xmml:name"/>s_scanswap_temp;
	<!-- end bug fix -->
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_<xsl:value-of select="$xagentName"/>, 
        temp_scan_storage_bytes_<xsl:value-of select="$xagentName"/>, 
        d_<xsl:value-of select="xmml:name"/>s_swap->_scan_input,
        d_<xsl:value-of select="xmml:name"/>s_swap->_position,
        h_xmachine_memory_<xsl:value-of select="$xagentName"/>_count, 
        stream
    );

	//Scatter
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, scatter_optional_<xsl:value-of select="xmml:name"/>_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_optional_<xsl:value-of select="xmml:name"/>_messages&lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_<xsl:value-of select="xmml:name"/>s, d_<xsl:value-of select="xmml:name"/>s_swap);
	gpuErrchkLaunch();
	</xsl:if></xsl:if>
	</xsl:for-each></xsl:if>
	</xsl:if>
	
	<xsl:if test="xmml:outputs/gpu:output"><xsl:variable name="messageName" select="xmml:outputs/gpu:output/xmml:messageName"/><xsl:variable name="outputType" select="xmml:outputs/gpu:output/gpu:type"/><xsl:variable name="xagentName" select="../../xmml:name"/>
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT <xsl:if test="../../gpu:type='continuous'">
	<xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messageName]">
  <xsl:if test="gpu:partitioningNone or gpu:partitioningSpatial or gpu:partitioningGraphEdge">
	<xsl:if test="$outputType='optional_message'">
	gpuErrchk( cudaMemcpy( &amp;scan_last_sum, &amp;d_<xsl:value-of select="xmml:name"/>s_swap->_position[h_xmachine_memory_<xsl:value-of select="$xagentName"/>_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &amp;scan_last_included, &amp;d_<xsl:value-of select="xmml:name"/>s_swap->_scan_input[h_xmachine_memory_<xsl:value-of select="$xagentName"/>_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//If last item in prefix sum was 1 then increase its index to get the count
	if (scan_last_included == 1){
		h_message_<xsl:value-of select="xmml:name"/>_count += scan_last_sum+1;
	}else{
		h_message_<xsl:value-of select="xmml:name"/>_count += scan_last_sum;
	}
    </xsl:if><xsl:if test="$outputType='single_message'">
	h_message_<xsl:value-of select="xmml:name"/>_count += h_xmachine_memory_<xsl:value-of select="$xagentName"/>_count;
	</xsl:if>//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_<xsl:value-of select="xmml:name"/>_count, &amp;h_message_<xsl:value-of select="xmml:name"/>_count, sizeof(int)));	
	</xsl:if>
	</xsl:for-each>
	</xsl:if>
	</xsl:if>

  <xsl:if test="xmml:xagentOutputs/gpu:xagentOutput">
	<xsl:variable name="xagent_output" select="xmml:xagentOutputs/gpu:xagentOutput/xmml:xagentName"/><xsl:if test="../../../gpu:xagent[xmml:name=$xagent_output]/gpu:type='continuous'">
    //COPY ANY AGENT COUNT BEFORE <xsl:value-of select="../../xmml:name"/> AGENTS ARE KILLED (needed for scatter)
	int <xsl:value-of select="../../xmml:name"/>s_pre_death_count = h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count;
	</xsl:if>
	</xsl:if>
	
	<xsl:if test="../../gpu:type='continuous'"><xsl:if test="gpu:reallocate='true'">
	//FOR CONTINUOUS AGENTS WITH REALLOCATION REMOVE POSSIBLE DEAD AGENTS	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_<xsl:value-of select="../../xmml:name"/>, 
        temp_scan_storage_bytes_<xsl:value-of select="../../xmml:name"/>, 
        d_<xsl:value-of select="../../xmml:name"/>s->_scan_input,
        d_<xsl:value-of select="../../xmml:name"/>s->_position,
        h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, 
        stream
    );

	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, scatter_<xsl:value-of select="../../xmml:name"/>_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_<xsl:value-of select="../../xmml:name"/>_Agents&lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s_swap, d_<xsl:value-of select="../../xmml:name"/>s, 0, h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count);
	gpuErrchkLaunch();
	//use a temp pointer to make swap default
	xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* <xsl:value-of select="xmml:name"/>_<xsl:value-of select="../../xmml:name"/>s_temp = d_<xsl:value-of select="../../xmml:name"/>s;
	d_<xsl:value-of select="../../xmml:name"/>s = d_<xsl:value-of select="../../xmml:name"/>s_swap;
	d_<xsl:value-of select="../../xmml:name"/>s_swap = <xsl:value-of select="xmml:name"/>_<xsl:value-of select="../../xmml:name"/>s_temp;
	//reset agent count
	gpuErrchk( cudaMemcpy( &amp;scan_last_sum, &amp;d_<xsl:value-of select="../../xmml:name"/>s_swap->_position[h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &amp;scan_last_included, &amp;d_<xsl:value-of select="../../xmml:name"/>s_swap->_scan_input[h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count = scan_last_sum+1;
	else
		h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count = scan_last_sum;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, sizeof(int)));	
	</xsl:if></xsl:if>

	<xsl:if test="xmml:xagentOutputs/gpu:xagentOutput"><xsl:for-each select="xmml:xagentOutputs/gpu:xagentOutput">
	<xsl:variable name="xagent_output" select="xmml:xagentName"/><xsl:if test="../../../../../gpu:xagent[xmml:name=$xagent_output]/gpu:type='continuous'">
	//FOR <xsl:value-of select="xmml:xagentName"/> AGENT OUTPUT SCATTER AGENTS 

    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_<xsl:value-of select="xmml:xagentName"/>, 
        temp_scan_storage_bytes_<xsl:value-of select="xmml:xagentName"/>, 
        d_<xsl:value-of select="xmml:xagentName"/>s_new->_scan_input, 
        d_<xsl:value-of select="xmml:xagentName"/>s_new->_position, 
        <xsl:value-of select="../../../../xmml:name"/>s_pre_death_count,
        stream
    );

	//reset agent count
	int <xsl:value-of select="xmml:xagentName"/>_after_birth_count;
	gpuErrchk( cudaMemcpy( &amp;scan_last_sum, &amp;d_<xsl:value-of select="xmml:xagentName"/>s_new->_position[<xsl:value-of select="../../../../xmml:name"/>s_pre_death_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &amp;scan_last_included, &amp;d_<xsl:value-of select="xmml:xagentName"/>s_new->_scan_input[<xsl:value-of select="../../../../xmml:name"/>s_pre_death_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		<xsl:value-of select="xmml:xagentName"/>_after_birth_count = h_xmachine_memory_<xsl:value-of select="xmml:xagentName"/>_<xsl:value-of select="xmml:state"/>_count + scan_last_sum+1;
	else
		<xsl:value-of select="xmml:xagentName"/>_after_birth_count = h_xmachine_memory_<xsl:value-of select="xmml:xagentName"/>_<xsl:value-of select="xmml:state"/>_count + scan_last_sum;
	//check buffer is not exceeded
	if (<xsl:value-of select="xmml:xagentName"/>_after_birth_count > xmachine_memory_<xsl:value-of select="xmml:xagentName"/>_MAX){
		printf("Error: Buffer size of <xsl:value-of select="xmml:xagentName"/> agents in state <xsl:value-of select="xmml:state"/> will be exceeded writing new agents in function <xsl:value-of select="../../xmml:name"/>\n");
		exit(EXIT_FAILURE);
	}
	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, scatter_<xsl:value-of select="xmml:xagentName"/>_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_<xsl:value-of select="xmml:xagentName"/>_Agents&lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_<xsl:value-of select="xmml:xagentName"/>s_<xsl:value-of select="xmml:state"/>, d_<xsl:value-of select="xmml:xagentName"/>s_new, h_xmachine_memory_<xsl:value-of select="xmml:xagentName"/>_<xsl:value-of select="xmml:state"/>_count, <xsl:value-of select="../../../../xmml:name"/>s_pre_death_count);
	gpuErrchkLaunch();
	//Copy count to device
	h_xmachine_memory_<xsl:value-of select="xmml:xagentName"/>_<xsl:value-of select="xmml:state"/>_count = <xsl:value-of select="xmml:xagentName"/>_after_birth_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="xmml:xagentName"/>_<xsl:value-of select="xmml:state"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="xmml:xagentName"/>_<xsl:value-of select="xmml:state"/>_count, sizeof(int)));	
	</xsl:if></xsl:for-each>
	</xsl:if>
	
	<xsl:if test="xmml:outputs/gpu:output"><xsl:variable name="messageName" select="xmml:outputs/gpu:output/xmml:messageName"/>
	<xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messageName]">
	<xsl:if test="gpu:partitioningSpatial">
	//reset partition matrix
	gpuErrchk( cudaMemset( (void*) d_<xsl:value-of select="xmml:name"/>_partition_matrix, 0, sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>_PBM)));
    //PR Bug fix: Second fix. This should prevent future problems when multiple agents write the same message as now the message structure is completely rebuilt after an output.
    if (h_message_<xsl:value-of select="xmml:name"/>_count > 0){
#ifdef FAST_ATOMIC_SORTING
      //USE ATOMICS TO BUILD PARTITION BOUNDARY
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, hist_<xsl:value-of select="xmml:name"/>_messages, no_sm, h_message_<xsl:value-of select="xmml:name"/>_count); 
	  gridSize = (h_message_<xsl:value-of select="xmml:name"/>_count + blockSize - 1) / blockSize;
	  hist_<xsl:value-of select="xmml:name"/>_messages&lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_xmachine_message_<xsl:value-of select="xmml:name"/>_local_bin_index, d_xmachine_message_<xsl:value-of select="xmml:name"/>_unsorted_index, d_<xsl:value-of select="xmml:name"/>_partition_matrix->end_or_count, d_<xsl:value-of select="xmml:name"/>s, h_message_<xsl:value-of select="xmml:name"/>_count);
	  gpuErrchkLaunch();
	
      // Scan
      cub::DeviceScan::ExclusiveSum(
          d_temp_scan_storage_xmachine_message_<xsl:value-of select="xmml:name"/>, 
          temp_scan_bytes_xmachine_message_<xsl:value-of select="xmml:name"/>, 
          d_<xsl:value-of select="xmml:name"/>_partition_matrix->end_or_count,
          d_<xsl:value-of select="xmml:name"/>_partition_matrix->start,
          xmachine_message_<xsl:value-of select="xmml:name"/>_grid_size, 
          stream
      );
	
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, reorder_<xsl:value-of select="xmml:name"/>_messages, no_sm, h_message_<xsl:value-of select="xmml:name"/>_count); 
	  gridSize = (h_message_<xsl:value-of select="xmml:name"/>_count + blockSize - 1) / blockSize; 	// Round up according to array size 
	  reorder_<xsl:value-of select="xmml:name"/>_messages &lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_xmachine_message_<xsl:value-of select="xmml:name"/>_local_bin_index, d_xmachine_message_<xsl:value-of select="xmml:name"/>_unsorted_index, d_<xsl:value-of select="xmml:name"/>_partition_matrix->start, d_<xsl:value-of select="xmml:name"/>s, d_<xsl:value-of select="xmml:name"/>s_swap, h_message_<xsl:value-of select="xmml:name"/>_count);
	  gpuErrchkLaunch();
#else
	  //HASH, SORT, REORDER AND BUILD PMB FOR SPATIAL PARTITIONING MESSAGE OUTPUTS
	  //Get message hash values for sorting
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, hash_<xsl:value-of select="xmml:name"/>_messages, no_sm, h_message_<xsl:value-of select="xmml:name"/>_count); 
	  gridSize = (h_message_<xsl:value-of select="xmml:name"/>_count + blockSize - 1) / blockSize;
	  hash_<xsl:value-of select="xmml:name"/>_messages&lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_xmachine_message_<xsl:value-of select="xmml:name"/>_keys, d_xmachine_message_<xsl:value-of select="xmml:name"/>_values, d_<xsl:value-of select="xmml:name"/>s);
    gpuErrchkLaunch();
    //Sort
    cub::DeviceRadixSort::SortPairs(d_CUB_temp_storage_<xsl:value-of select="xmml:name"/>, CUB_temp_storage_bytes_<xsl:value-of select="xmml:name"/>, d_xmachine_message_<xsl:value-of select="xmml:name"/>_keys, d_xmachine_message_<xsl:value-of select="xmml:name"/>_keys_swap, d_xmachine_message_<xsl:value-of select="xmml:name"/>_values, d_xmachine_message_<xsl:value-of select="xmml:name"/>_values_swap, h_message_<xsl:value-of select="xmml:name"/>_count, 0, binCountBits_<xsl:value-of select="xmml:name"/>, stream);
    {
    unsigned int *_t = d_xmachine_message_<xsl:value-of select="xmml:name"/>_keys;
    d_xmachine_message_<xsl:value-of select="xmml:name"/>_keys = d_xmachine_message_<xsl:value-of select="xmml:name"/>_keys_swap;
    d_xmachine_message_<xsl:value-of select="xmml:name"/>_keys_swap = _t;
    }
    {
    unsigned int *_t = d_xmachine_message_<xsl:value-of select="xmml:name"/>_values;
    d_xmachine_message_<xsl:value-of select="xmml:name"/>_values = d_xmachine_message_<xsl:value-of select="xmml:name"/>_values_swap;
    d_xmachine_message_<xsl:value-of select="xmml:name"/>_values_swap = _t;
    }
    gpuErrchkLaunch();
    //reorder and build pcb
    gpuErrchk(cudaMemset(d_<xsl:value-of select="xmml:name"/>_partition_matrix->start, 0xffffffff, xmachine_message_<xsl:value-of select="xmml:name"/>_grid_size* sizeof(int)));
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, reorder_<xsl:value-of select="xmml:name"/>_messages, reorder_messages_sm_size, h_message_<xsl:value-of select="xmml:name"/>_count); 
	  gridSize = (h_message_<xsl:value-of select="xmml:name"/>_count + blockSize - 1) / blockSize;
	  int reorder_sm_size = reorder_messages_sm_size(blockSize);
	  reorder_<xsl:value-of select="xmml:name"/>_messages&lt;&lt;&lt;gridSize, blockSize, reorder_sm_size, stream&gt;&gt;&gt;(d_xmachine_message_<xsl:value-of select="xmml:name"/>_keys, d_xmachine_message_<xsl:value-of select="xmml:name"/>_values, d_<xsl:value-of select="xmml:name"/>_partition_matrix, d_<xsl:value-of select="xmml:name"/>s, d_<xsl:value-of select="xmml:name"/>s_swap);
	  gpuErrchkLaunch();
#endif
  }
	//swap ordered list
	xmachine_message_<xsl:value-of select="xmml:name"/>_list* d_<xsl:value-of select="xmml:name"/>s_temp = d_<xsl:value-of select="xmml:name"/>s;
	d_<xsl:value-of select="xmml:name"/>s = d_<xsl:value-of select="xmml:name"/>s_swap;
	d_<xsl:value-of select="xmml:name"/>s_swap = d_<xsl:value-of select="xmml:name"/>s_temp;
	</xsl:if>


<xsl:if test="gpu:partitioningGraphEdge">
  // Sort messages based on the edge index, and construct the relevant data structure for graph edge based messaging. Keys are sorted and then message data is scattered. 

  // Reset the message bounds data structure to 0
  gpuErrchk(cudaMemset((void*)d_xmachine_message_<xsl:value-of select="xmml:name"/>_bounds, 0, sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>_bounds)));

  // If there are any messages output (to account for 0 optional messages)
  if (h_message_<xsl:value-of select="xmml:name"/>_count > 0){
  // Build histogram using atomics
  cudaOccupancyMaxPotentialBlockSizeVariableSMem(&amp;minGridSize, &amp;blockSize, hist_<xsl:value-of select="xmml:name"/>_messages, no_sm, h_message_<xsl:value-of select="xmml:name"/>_count);
  gridSize = (h_message_<xsl:value-of select="xmml:name"/>_count + blockSize - 1) / blockSize;
  hist_<xsl:value-of select="xmml:name"/>_messages &lt;&lt;&lt;gridSize, blockSize, 0, stream &gt;&gt;&gt;(d_xmachine_message_<xsl:value-of select="xmml:name"/>_scatterer-&gt;edge_local_index, d_xmachine_message_<xsl:value-of select="xmml:name"/>_scatterer-&gt;unsorted_edge_index, d_xmachine_message_<xsl:value-of select="xmml:name"/>_bounds-&gt;count, d_<xsl:value-of select="xmml:name"/>s, h_message_<xsl:value-of select="xmml:name"/>_count);
  gpuErrchkLaunch();

  // Exclusive scan on histogram output to find the index for each message for each edge/bucket
  cub::DeviceScan::ExclusiveSum(
      d_temp_scan_storage_xmachine_message_<xsl:value-of select="xmml:name"/>,
      temp_scan_bytes_xmachine_message_<xsl:value-of select="xmml:name"/>,
      d_xmachine_message_<xsl:value-of select="xmml:name"/>_bounds-&gt;count,
      d_xmachine_message_<xsl:value-of select="xmml:name"/>_bounds-&gt;start,
      staticGraph_<xsl:value-of select="gpu:partitioningGraphEdge/gpu:environmentGraph"/>_edge_bufferSize, 
      stream
  );
  gpuErrchkLaunch();

  // Launch kernel to re-order (scatter) the messages
  cudaOccupancyMaxPotentialBlockSizeVariableSMem(&amp;minGridSize, &amp;blockSize, reorder_<xsl:value-of select="xmml:name"/>_messages, no_sm, h_message_<xsl:value-of select="xmml:name"/>_count);
  gridSize = (h_message_<xsl:value-of select="xmml:name"/>_count + blockSize - 1) / blockSize;  // Round up according to array size
  reorder_<xsl:value-of select="xmml:name"/>_messages &lt;&lt;&lt;gridSize, blockSize, 0, stream &gt;&gt;&gt;(d_xmachine_message_<xsl:value-of select="xmml:name"/>_scatterer-&gt;edge_local_index, d_xmachine_message_<xsl:value-of select="xmml:name"/>_scatterer-&gt;unsorted_edge_index, d_xmachine_message_<xsl:value-of select="xmml:name"/>_bounds-&gt;start, d_<xsl:value-of select="xmml:name"/>s, d_<xsl:value-of select="xmml:name"/>s_swap, h_message_<xsl:value-of select="xmml:name"/>_count);
  gpuErrchkLaunch();
  }
  // Pointer swap the double buffers.
  xmachine_message_<xsl:value-of select="xmml:name"/>_list* d_<xsl:value-of select="xmml:name"/>s_temp = d_<xsl:value-of select="xmml:name"/>s;
  d_<xsl:value-of select="xmml:name"/>s = d_<xsl:value-of select="xmml:name"/>s_swap;
  d_<xsl:value-of select="xmml:name"/>s_swap = d_<xsl:value-of select="xmml:name"/>s_temp;

  </xsl:if>

	</xsl:for-each>
	</xsl:if>
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    <xsl:choose>
    <xsl:when test="../../gpu:type='continuous'">
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:nextState"/>_count+h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count > xmachine_memory_<xsl:value-of select="../../xmml:name"/>_MAX){
		printf("Error: Buffer size of <xsl:value-of select="xmml:name"/> agents in state <xsl:value-of select="xmml:nextState"/> will be exceeded moving working agents to next state in function <xsl:value-of select="xmml:name"/>\n");
      exit(EXIT_FAILURE);
      }
      <xsl:choose>
        <xsl:when test="xmml:currentState=xmml:nextState and not(xmml:condition) and not(gpu:globalCondition) and gpu:reallocate='false' and not(xmml:xagentOutputs/gpu:xagentOutput)">
  //pointer swap the updated data
  <xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>_temp = d_<xsl:value-of select="../../xmml:name"/>s;
  d_<xsl:value-of select="../../xmml:name"/>s = d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>;
  d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/> = <xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>_temp;
        </xsl:when>
        <xsl:otherwise>
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &amp;minGridSize, &amp;blockSize, append_<xsl:value-of select="../../xmml:name"/>_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_<xsl:value-of select="../../xmml:name"/>_Agents&lt;&lt;&lt;gridSize, blockSize, 0, stream&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:nextState"/>, d_<xsl:value-of select="../../xmml:name"/>s, h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:nextState"/>_count, h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count);
  gpuErrchkLaunch();
        </xsl:otherwise>
      </xsl:choose>
	//update new state agent size
	h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:nextState"/>_count += h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:nextState"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:nextState"/>_count, sizeof(int)));	
	</xsl:when>
    <xsl:when test="../../gpu:type='discrete'">
    //currentState maps to working list
	<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>_temp = d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>;
	d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/> = d_<xsl:value-of select="../../xmml:name"/>s;
	d_<xsl:value-of select="../../xmml:name"/>s = <xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>_temp;
    //set current state count
	h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count = h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count, sizeof(int)));	
	</xsl:when>
  </xsl:choose>
	
}


</xsl:for-each>
    

<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state"> 
extern void reset_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count()
{
    h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count = 0;
}
</xsl:for-each>
    
</xsl:template>
</xsl:stylesheet>
