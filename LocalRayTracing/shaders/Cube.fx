//--------------------------------------------------------------------------------------
// File: Cube.fx
//--------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------
// Includes
//--------------------------------------------------------------------------------------
#include "shared.fxh"


//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------
float4x4	g_mWorld;						// World matrix for cube


//--------------------------------------------------------------------------------------
// Vertex shader output structure
//--------------------------------------------------------------------------------------
struct VS_OUTPUT
{
    float4 Position		: POSITION;   // vertex position 
    float3 TextureXYZ	: TEXCOORD0;  // vertex texture coords 
};


//--------------------------------------------------------------------------------------
// This shader computes standard transform
//--------------------------------------------------------------------------------------
VS_OUTPUT Transform_VS(float3 vPos : POSITION)
{
	VS_OUTPUT Output;
	
	float4x4 g_mWorldViewProjection = mul(g_mWorld, g_mViewProjection);
	
	// Transform the position from object space to homogeneous projection space
	Output.Position = mul(float4(vPos, 1), g_mWorldViewProjection);
	
	// Just copy the texture coordinate through
	Output.TextureXYZ = vPos;
	
	return Output;
}


//--------------------------------------------------------------------------------------
// Cube pixel shader
//--------------------------------------------------------------------------------------
float4 Cube_PS(VS_OUTPUT In) : COLOR0
{
	float4 Output;
	
	// Lookup cube map
	Output.rgb = texCUBE(CubeMapSampler, In.TextureXYZ);
	
	Output.a = 1;
	
	return Output;
}


//--------------------------------------------------------------------------------------
// Techniques
//--------------------------------------------------------------------------------------
technique Cube
{
	pass P0
	{
		VertexShader	= compile vs_2_0 Transform_VS();
		PixelShader		= compile ps_2_0 Cube_PS();
		CullMode		= CW;
	}
}