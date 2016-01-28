//--------------------------------------------------------------------------------------
// File: shared.fxh
//--------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------
// Shared variables
//--------------------------------------------------------------------------------------
shared textureCUBE	g_CubeTexture;			// color texture for cube

shared float		g_fTime;				// App's time in seconds
shared float4x4		g_mViewProjection;		// View * Projection matrix


//--------------------------------------------------------------------------------------
// Texture samplers
//--------------------------------------------------------------------------------------
samplerCUBE CubeMapSampler = 
sampler_state
{
    Texture = <g_CubeTexture>;
    MipFilter = LINEAR;
    MinFilter = ANISOTROPIC;
    MagFilter = LINEAR;
    MaxAnisotropy = 16;
    AddressU = CLAMP;
    AddressV = CLAMP;
    AddressW = CLAMP;
};
