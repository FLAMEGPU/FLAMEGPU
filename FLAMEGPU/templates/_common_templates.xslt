<?xml version="1.0" encoding="utf-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:xmml="http://www.dcs.shef.ac.uk/~paul/XMML" xmlns:gpu="http://www.dcs.shef.ac.uk/~paul/XMMLGPU">

<xsl:template name="copyrightNotice">
/*
 * FLAME GPU v 1.5.X for CUDA 9
 * Copyright University of Sheffield.
 * Original Author: Dr Paul Richmond (user contributions tracked on https://github.com/FLAMEGPU/FLAMEGPU)
 * Contact: p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
 *
 * University of Sheffield retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * University of Sheffield is strictly prohibited.
 *
 * For terms of licence agreement please attached licence or view licence
 * on www.flamegpu.com website.
 *
 */
</xsl:template>


<!-- format specifier template function -->
<xsl:template name="formatSpecifier">
    <xsl:param name="type"/>
    <xsl:choose>
        <xsl:when test="$type='char'">%d</xsl:when>
        <xsl:when test="$type='unsigned char'">%u</xsl:when>
        <xsl:when test="$type='short'">%d</xsl:when>
        <xsl:when test="$type='unsigned short'">%u</xsl:when>
        <xsl:when test="$type='int'">%d</xsl:when>
        <xsl:when test="$type='unsigned int'">%u</xsl:when>
        <xsl:when test="$type='long long int'">%lld</xsl:when>
        <xsl:when test="$type='unsigned long long int'">%llu</xsl:when>
        
        <xsl:when test="$type='ivec2'">%d, %d</xsl:when>
        <xsl:when test="$type='uvec2'">%u, %u</xsl:when>
        <xsl:when test="$type='fvec2'">%f, %f</xsl:when>
        <xsl:when test="$type='dvec2'">%f, %f</xsl:when>
        
        <xsl:when test="$type='ivec3'">%d, %d, %d</xsl:when>
        <xsl:when test="$type='uvec3'">%u, %u, %u</xsl:when>
        <xsl:when test="$type='fvec3'">%f, %f, %f</xsl:when>
        <xsl:when test="$type='dvec3'">%f, %f, %f</xsl:when>
        
        <xsl:when test="$type='ivec4'">%d, %d, %d, %d</xsl:when>
        <xsl:when test="$type='uvec4'">%u, %u, %u, %u</xsl:when>
        <xsl:when test="$type='fvec4'">%f, %f, %f, %f</xsl:when>
        <xsl:when test="$type='dvec4'">%f, %f, %f, %f</xsl:when>
        
        <xsl:otherwise>%f</xsl:otherwise> <!-- default output format is float -->
    </xsl:choose>
</xsl:template>
  
<!-- Default variable initialiser with optional default value argument -->
<xsl:template name="defaultInitialiser">
    <xsl:param name="type"/>
    <xsl:param name="defaultValue"/>
    <xsl:choose>
        <xsl:when test="$type='ivec2'">{<xsl:choose><xsl:when test="$defaultValue"><xsl:value-of select="$defaultValue"/></xsl:when><xsl:otherwise>0, 0</xsl:otherwise></xsl:choose>}</xsl:when>
        <xsl:when test="$type='uvec2'">{<xsl:choose><xsl:when test="$defaultValue"><xsl:value-of select="$defaultValue"/></xsl:when><xsl:otherwise>0, 0</xsl:otherwise></xsl:choose>}</xsl:when>
        <xsl:when test="$type='fvec2'">{<xsl:choose><xsl:when test="$defaultValue"><xsl:value-of select="$defaultValue"/></xsl:when><xsl:otherwise>0, 0</xsl:otherwise></xsl:choose>}</xsl:when>
        <xsl:when test="$type='dvec2'">{<xsl:choose><xsl:when test="$defaultValue"><xsl:value-of select="$defaultValue"/></xsl:when><xsl:otherwise>0, 0</xsl:otherwise></xsl:choose>}</xsl:when>
        
        <xsl:when test="$type='ivec3'">{<xsl:choose><xsl:when test="$defaultValue"><xsl:value-of select="$defaultValue"/></xsl:when><xsl:otherwise>0, 0, 0</xsl:otherwise></xsl:choose>}</xsl:when>
        <xsl:when test="$type='uvec3'">{<xsl:choose><xsl:when test="$defaultValue"><xsl:value-of select="$defaultValue"/></xsl:when><xsl:otherwise>0, 0, 0</xsl:otherwise></xsl:choose>}</xsl:when>
        <xsl:when test="$type='fvec3'">{<xsl:choose><xsl:when test="$defaultValue"><xsl:value-of select="$defaultValue"/></xsl:when><xsl:otherwise>0, 0, 0</xsl:otherwise></xsl:choose>}</xsl:when>
        <xsl:when test="$type='dvec3'">{<xsl:choose><xsl:when test="$defaultValue"><xsl:value-of select="$defaultValue"/></xsl:when><xsl:otherwise>0, 0, 0</xsl:otherwise></xsl:choose>}</xsl:when>
        
        <xsl:when test="$type='ivec4'">{<xsl:choose><xsl:when test="$defaultValue"><xsl:value-of select="$defaultValue"/></xsl:when><xsl:otherwise>0, 0, 0, 0</xsl:otherwise></xsl:choose>}</xsl:when>
        <xsl:when test="$type='uvec4'">{<xsl:choose><xsl:when test="$defaultValue"><xsl:value-of select="$defaultValue"/></xsl:when><xsl:otherwise>0, 0, 0, 0</xsl:otherwise></xsl:choose>}</xsl:when>
        <xsl:when test="$type='fvec4'">{<xsl:choose><xsl:when test="$defaultValue"><xsl:value-of select="$defaultValue"/></xsl:when><xsl:otherwise>0, 0, 0, 0</xsl:otherwise></xsl:choose>}</xsl:when>
        <xsl:when test="$type='dvec4'">{<xsl:choose><xsl:when test="$defaultValue"><xsl:value-of select="$defaultValue"/></xsl:when><xsl:otherwise>0, 0, 0, 0</xsl:otherwise></xsl:choose>}</xsl:when>
        
        <!-- default output format is float -->
        <xsl:otherwise>
            <xsl:choose><xsl:when test="$defaultValue"><xsl:value-of select="$defaultValue"/></xsl:when><xsl:otherwise>0</xsl:otherwise></xsl:choose>
        </xsl:otherwise>
    </xsl:choose>
</xsl:template>



<!-- argument list generator for agent variable outputs -->
<xsl:template name="outputVariable">
    <xsl:param name="agent_name"/>
    <xsl:param name="state_name"/>
    <xsl:param name="variable_name"/>
    <xsl:param name="variable_type"/>
    <xsl:choose>      
        <xsl:when test="contains($variable_type, '2')">h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state_name"/>-><xsl:value-of select="$variable_name"/>[i].x, h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state_name"/>-><xsl:value-of select="$variable_name"/>[i].y</xsl:when>
        <xsl:when test="contains($variable_type, '3')">h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state_name"/>-><xsl:value-of select="$variable_name"/>[i].x, h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state_name"/>-><xsl:value-of select="$variable_name"/>[i].y, h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state_name"/>-><xsl:value-of select="$variable_name"/>[i].z</xsl:when>
        <xsl:when test="contains($variable_type, '4')">h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state_name"/>-><xsl:value-of select="$variable_name"/>[i].x, h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state_name"/>-><xsl:value-of select="$variable_name"/>[i].y, h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state_name"/>-><xsl:value-of select="$variable_name"/>[i].z, h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state_name"/>-><xsl:value-of select="$variable_name"/>[i].w</xsl:when>
        <xsl:otherwise>h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state_name"/>-><xsl:value-of select="$variable_name"/>[i]</xsl:otherwise> <!-- default output format is scalar type -->
    </xsl:choose>
</xsl:template>
  
<!-- argument list generator for agent variable array outputs --> 
<xsl:template name="outputVariableArrayItem">
    <xsl:param name="agent_name"/>
    <xsl:param name="state_name"/>
    <xsl:param name="variable_name"/>
    <xsl:param name="variable_type"/>
    <xsl:choose>      
        <xsl:when test="contains($variable_type, '2')">h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state_name"/>-><xsl:value-of select="$variable_name"/>[(j*xmachine_memory_<xsl:value-of select="$agent_name"/>_MAX)+i].x, h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state_name"/>-><xsl:value-of select="$variable_name"/>[(j*xmachine_memory_<xsl:value-of select="$agent_name"/>_MAX)+i].y</xsl:when>
        <xsl:when test="contains($variable_type, '3')">h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state_name"/>-><xsl:value-of select="$variable_name"/>[(j*xmachine_memory_<xsl:value-of select="$agent_name"/>_MAX)+i].x, h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state_name"/>-><xsl:value-of select="$variable_name"/>[(j*xmachine_memory_<xsl:value-of select="$agent_name"/>_MAX)+i].y, h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state_name"/>-><xsl:value-of select="$variable_name"/>[(j*xmachine_memory_<xsl:value-of select="$agent_name"/>_MAX)+i].z</xsl:when>
        <xsl:when test="contains($variable_type, '4')">h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state_name"/>-><xsl:value-of select="$variable_name"/>[(j*xmachine_memory_<xsl:value-of select="$agent_name"/>_MAX)+i].x, h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state_name"/>-><xsl:value-of select="$variable_name"/>[(j*xmachine_memory_<xsl:value-of select="$agent_name"/>_MAX)+i].y, h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state_name"/>-><xsl:value-of select="$variable_name"/>[(j*xmachine_memory_<xsl:value-of select="$agent_name"/>_MAX)+i].z, h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state_name"/>-><xsl:value-of select="$variable_name"/>[(j*xmachine_memory_<xsl:value-of select="$agent_name"/>_MAX)+i].w</xsl:when>
        <xsl:otherwise>h_<xsl:value-of select="$agent_name"/>s_<xsl:value-of select="$state_name"/>-><xsl:value-of select="$variable_name"/>[(j*xmachine_memory_<xsl:value-of select="$agent_name"/>_MAX)+i]</xsl:otherwise> <!-- default output format is scalar type -->
    </xsl:choose>
</xsl:template>

<!-- argument list generator for agent variable outputs -->
<xsl:template name="outputEnvironmentConstant">
    <xsl:param name="constant_name"/>
    <xsl:param name="constant_type"/>
    <xsl:choose>      
        <xsl:when test="contains($constant_type, '2')">(*get_<xsl:value-of select="$constant_name"/>()).x, (*get_<xsl:value-of select="$constant_name"/>()).y</xsl:when>
        <xsl:when test="contains($constant_type, '3')">(*get_<xsl:value-of select="$constant_name"/>()).x, (*get_<xsl:value-of select="$constant_name"/>()).y, (*get_<xsl:value-of select="$constant_name"/>()).z</xsl:when>
        <xsl:when test="contains($constant_type, '4')">(*get_<xsl:value-of select="$constant_name"/>()).x, (*get_<xsl:value-of select="$constant_name"/>()).y, (*get_<xsl:value-of select="$constant_name"/>()).z, (*get_<xsl:value-of select="$constant_name"/>()).w</xsl:when>
        <xsl:otherwise>(*get_<xsl:value-of select="$constant_name"/>())</xsl:otherwise> <!-- default output format is scalar type -->
    </xsl:choose>
</xsl:template>



<!-- argument list generator for environment constant array outputs --> 
<xsl:template name="outputEnvironmentConstantArrayItem">
    <xsl:param name="constant_name"/>
    <xsl:param name="constant_type"/>
    <xsl:choose>      
        <xsl:when test="contains($constant_type, '2')">get_<xsl:value-of select="$constant_name"/>()[j].x, get_<xsl:value-of select="$constant_name"/>()[j].y</xsl:when>
        <xsl:when test="contains($constant_type, '3')">get_<xsl:value-of select="$constant_name"/>()[j].x, get_<xsl:value-of select="$constant_name"/>()[j].y, get_<xsl:value-of select="$constant_name"/>()[j].z</xsl:when>
        <xsl:when test="contains($constant_type, '4')">get_<xsl:value-of select="$constant_name"/>()[j].x, get_<xsl:value-of select="$constant_name"/>()[j].y, get_<xsl:value-of select="$constant_name"/>()[j].z, get_<xsl:value-of select="$constant_name"/>()[j].w</xsl:when>
        <xsl:otherwise>get_<xsl:value-of select="$constant_name"/>()[j]</xsl:otherwise> <!-- default output format is scalar type -->
    </xsl:choose>
</xsl:template>

<!-- function pointer for reading variable types from string --> 
<xsl:template name="typeParserFunc">
    <xsl:param name="type"/>
    <xsl:choose>      
        <xsl:when test="$type='char'">fpgu_strtol</xsl:when>
        <xsl:when test="$type='unsigned char'">fpgu_strtoul</xsl:when>
        <xsl:when test="$type='short'">fpgu_strtol</xsl:when>
        <xsl:when test="$type='unsigned short'">fpgu_strtoul</xsl:when>
        <xsl:when test="$type='int'">fpgu_strtol</xsl:when>
        <xsl:when test="$type='unsigned int'">fpgu_strtoul</xsl:when>
        <xsl:when test="$type='long long int'">fpgu_strtoll</xsl:when>
        <xsl:when test="$type='unsigned long long int'">fpgu_strtoull</xsl:when> 
        <xsl:when test="$type='double'">fpgu_strtod</xsl:when>
        <xsl:when test="$type='float'">fgpu_atof</xsl:when>
      
        <xsl:when test="$type='ivec2'">fpgu_strtol</xsl:when>
        <xsl:when test="$type='uvec2'">fpgu_strtoul</xsl:when>
        <xsl:when test="$type='fvec2'">fgpu_atof</xsl:when>
        <xsl:when test="$type='dvec2'">fpgu_strtod</xsl:when>
        
        <xsl:when test="$type='ivec3'">fpgu_strtol</xsl:when>
        <xsl:when test="$type='uvec3'">fpgu_strtoul</xsl:when>
        <xsl:when test="$type='fvec3'">fgpu_atof</xsl:when>
        <xsl:when test="$type='dvec3'">fpgu_strtod</xsl:when>
      
        <xsl:when test="$type='ivec4'">fpgu_strtol</xsl:when>
        <xsl:when test="$type='uvec4'">fpgu_strtoul</xsl:when>
        <xsl:when test="$type='fvec4'">fgpu_atof</xsl:when>
        <xsl:when test="$type='dvec4'">fpgu_strtod</xsl:when>
      
        <xsl:otherwise>atof</xsl:otherwise> <!-- default parse function as float -->
    </xsl:choose>
</xsl:template>
  
  <!-- function pointer for reading variable types from string --> 
<xsl:template name="vectorBaseType">
    <xsl:param name="type"/>
    <xsl:choose>        
        <xsl:when test="$type='ivec2'">int</xsl:when>
        <xsl:when test="$type='uvec2'">unsigned int</xsl:when>
        <xsl:when test="$type='fvec2'">float</xsl:when>
        <xsl:when test="$type='dvec2'">double</xsl:when>
        
        <xsl:when test="$type='ivec3'">int</xsl:when>
        <xsl:when test="$type='uvec3'">unsigned int</xsl:when>
        <xsl:when test="$type='fvec3'">float</xsl:when>
        <xsl:when test="$type='dvec3'">double</xsl:when>
      
        <xsl:when test="$type='ivec4'">int</xsl:when>
        <xsl:when test="$type='uvec4'">unsigned int</xsl:when>
        <xsl:when test="$type='fvec4'">float</xsl:when>
        <xsl:when test="$type='dvec4'">double</xsl:when>
      
        <xsl:otherwise>float</xsl:otherwise> <!-- default base type of float -->
    </xsl:choose>
</xsl:template>



<!--Recursive template for function conditions-->
<xsl:template match="xmml:condition">(<xsl:choose>
<xsl:when test="xmml:lhs/xmml:value"><xsl:value-of select="xmml:lhs/xmml:value"/>
</xsl:when>
<xsl:when test="xmml:lhs/xmml:agentVariable">currentState-><xsl:value-of select="xmml:lhs/xmml:agentVariable"/>[index]</xsl:when>
<xsl:otherwise><xsl:apply-templates select="xmml:lhs/xmml:condition"/>
</xsl:otherwise>
</xsl:choose>
<xsl:value-of select="xmml:operator"/>
<xsl:choose>
<xsl:when test="xmml:rhs/xmml:value"><xsl:value-of select="xmml:rhs/xmml:value"/>
</xsl:when>
<xsl:when test="xmml:rhs/xmml:agentVariable">currentState-><xsl:value-of select="xmml:rhs/xmml:agentVariable"/>[index]</xsl:when>
<xsl:otherwise><xsl:apply-templates select="xmml:rhs/xmml:condition"/>
</xsl:otherwise>
</xsl:choose>)</xsl:template>

<!--Recursive template for function global conditions-->
<xsl:template match="gpu:globalCondition">(<xsl:choose>
<xsl:when test="xmml:lhs/xmml:value"><xsl:value-of select="xmml:lhs/xmml:value"/>
</xsl:when>
<xsl:when test="xmml:lhs/xmml:agentVariable">currentState-><xsl:value-of select="xmml:lhs/xmml:agentVariable"/>[index]</xsl:when>
<xsl:otherwise><xsl:apply-templates select="xmml:lhs/xmml:condition"/>
</xsl:otherwise>
</xsl:choose>
<xsl:value-of select="xmml:operator"/>
<xsl:choose>
<xsl:when test="xmml:rhs/xmml:value"><xsl:value-of select="xmml:rhs/xmml:value"/>
</xsl:when>
<xsl:when test="xmml:rhs/xmml:agentVariable">currentState-><xsl:value-of select="xmml:rhs/xmml:agentVariable"/>[index]</xsl:when>
<xsl:otherwise><xsl:apply-templates select="xmml:rhs/xmml:condition"/>
</xsl:otherwise>
</xsl:choose>)</xsl:template>

</xsl:stylesheet>
