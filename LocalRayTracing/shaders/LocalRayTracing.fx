//--------------------------------------------------------------------------------------
// File: LocalRayTracing.fx
//--------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------
// Includes
//--------------------------------------------------------------------------------------
#include "shared.fxh"


//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------
static const float3	g_MaterialAmbientColor  = { 0.2, 0.2, 0.2 };	// Material's ambient color
static const float3	g_MaterialDiffuseColor  = { 0.5, 0.5, 0.5 };	// Material's diffuse color
static const float3	g_MaterialSpecularColor = { 0.7, 0.7, 0.7 };	// Material's specular color
static const float	g_Shine = 128;
float		g_reflectance;
bool		g_reflections;
bool		g_cast_shadows;
float3		g_Eye;							// Eye position in world space
float3		g_WorldLightDir;				// Normalized light's direction in world space
float4		g_LightDiffuse;					// Light's diffuse color
int			g_vertex_count;					// Mesh vertex count
int			g_face_count;					// Mesh face count

texture2D	g_ColorTexture;					// Color texture for object
texture1D	g_position_texture;				// Texture with vertex positions
texture1D	g_texcoord_texture;				// Texture with vertex texcoords

float4x4	g_mWorld;						// World matrix for object

/*float3		PositionArray[12] =
{
	float3(-1.f, -2.f, 0),
	float3(+1.f, -2.f, 0),
	float3(-1.f, 0, 0),
	
	float3(-1.f, 0, 0),
	float3(+1.f, -2.f, 0),
	float3(+1.f, 0, 0),

	float3(-1.f, 0, 0),
	float3(+1.f, 0, 0),
	float3(-1.f, 0, -2.f),

	float3(-1.f, 0, -2.f),
	float3(+1.f, 0, 0),
	float3(+1.f, 0, -2.f)
};

float2		TexcoordArray[12] =
{
	float2(0.f, 0.f),
	float2(1.f, 0.f),
	float2(0.f, 1.f),

	float2(0.f, 1.f),
	float2(1.f, 0.f),
	float2(1.f, 1.f),

	float2(0.f, 1.f),
	float2(1.f, 1.f),
	float2(0.f, 0.f),

	float2(0.f, 0.f),
	float2(1.f, 1.f),
	float2(1.f, 0.f)
};*/


//--------------------------------------------------------------------------------------
// Texture samplers
//--------------------------------------------------------------------------------------
sampler2D ColorTextureSampler = sampler_state
{
    Texture = <g_ColorTexture>;
    MipFilter = LINEAR;
    MinFilter = ANISOTROPIC;
    MagFilter = LINEAR;
    MaxAnisotropy = 16;
};


sampler1D PositionSampler = sampler_state
{
	Texture = <g_position_texture>;
	MipFilter = NONE;
	MinFilter = POINT;
	MagFilter = POINT;
};


sampler1D TexcoordSampler = sampler_state
{
	Texture = <g_texcoord_texture>;
	MipFilter = NONE;
	MinFilter = POINT;
	MagFilter = POINT;
};


//--------------------------------------------------------------------------------------
// Vertex shaders output structures
//--------------------------------------------------------------------------------------
struct CUBEMAP_VS_OUTPUT
{
    float4	Position	: POSITION;		// vertex position 
    float2	TextureUV	: TEXCOORD0;	// vertex texture coords 
    float3	WorldPos	: TEXCOORD1;	// vertex position in world space
    float3	Normal		: TEXCOORD2;	// vertex normal in world space
    float3	Reflect		: TEXCOORD3;	// vertex reflection direction in world space
    int		FaceID		: TEXCOORD4;	// face id
};


struct RAYTRACE_VS_OUTPUT
{
    float4	Position		: POSITION;		// vertex position 
    float2	TextureUV		: TEXCOORD0;	// vertex texture coords 
    float3	ObjPos			: TEXCOORD1;	// vertex position in object space
    float3	WorldPos		: TEXCOORD2;	// vertex position in world space
    float3	Normal			: TEXCOORD3;	// vertex normal in world space
    float3	ObjReflect		: TEXCOORD4;	// vertex reflection direction in object space
    float3	WorldReflect	: TEXCOORD5;	// vertex reflection direction in world space
    int		FaceID			: TEXCOORD6;	// face id
};


//--------------------------------------------------------------------------------------
// Cube map vertex shader
//--------------------------------------------------------------------------------------
CUBEMAP_VS_OUTPUT CubeMap_VS(float3 vPos : POSITION, float3 vNormal : NORMAL, float2 vTexCoord0 : TEXCOORD0, int vFaceID : TEXCOORD1)
{
	CUBEMAP_VS_OUTPUT Output;
	
	float4x4 g_mWorldViewProjection = mul(g_mWorld, g_mViewProjection);
	
	// Transform the position from object space to homogeneous projection space
	Output.Position = mul(float4(vPos, 1), g_mWorldViewProjection);
	
	// Just copy the texture coordinate through
	Output.TextureUV = vTexCoord0;
	
	// Transform the position from object space to world space
	Output.WorldPos = mul(float4(vPos, 1), g_mWorld);

	// view direction
	float3 v = Output.WorldPos - g_Eye;
	
	// Transform the normal from object space to world space
	Output.Normal = normalize(mul(vNormal, (float3x3)g_mWorld)); // normal (world space)
	
	// Flip normal if necessary
	Output.Normal = faceforward(Output.Normal, v, Output.Normal);
	
	// Find reflection direction in world space
	Output.Reflect = reflect(v, Output.Normal);
	
	// Just copy the face id through
	Output.FaceID = vFaceID;
	
	return Output;
}


//--------------------------------------------------------------------------------------
// Ray tracing vertex shader
//--------------------------------------------------------------------------------------
RAYTRACE_VS_OUTPUT Raytrace_VS(float3 vPos : POSITION, float3 vNormal : NORMAL, float2 vTexCoord0 : TEXCOORD0, int vFaceID : TEXCOORD1)
{
	RAYTRACE_VS_OUTPUT Output;
	
	float4x4 g_mWorldViewProjection = mul(g_mWorld, g_mViewProjection);
	
	// Transform the position from object space to homogeneous projection space
	Output.Position = mul(float4(vPos, 1), g_mWorldViewProjection);
	
	// Just copy the texture coordinate through
	Output.TextureUV = vTexCoord0;
	
	// Just copy the object position through
	Output.ObjPos = vPos;
	
	// Transform the position from object space to world space
	Output.WorldPos = mul(float4(vPos, 1), g_mWorld);

	// view direction
	float3 v = Output.WorldPos - g_Eye;
	
	// Transform the normal from object space to world space
	Output.Normal = normalize(mul(vNormal, (float3x3)g_mWorld)); // normal (world space)
	
	// Flip normal if necessary
	Output.Normal = faceforward(Output.Normal, v, Output.Normal);
	
	// Find reflection direction in world space
	Output.WorldReflect = reflect(v, Output.Normal);
	
	// Transform reflection direction in object space
	Output.ObjReflect = mul((float3x3)g_mWorld, Output.WorldReflect);
	
	// Just copy the face id through
	Output.FaceID = vFaceID;
	
	return Output;
}


//--------------------------------------------------------------------------------------
// This function compute Blinn-Phong lighting
//--------------------------------------------------------------------------------------
float3 BlinnPhong(float3 TexColor, float3 n, float3 v, float3 l, float shadow = 1)
{
	float3 h = normalize(v + l);
	float3 color = g_MaterialDiffuseColor * TexColor * saturate(dot(l, n));
	color += g_MaterialSpecularColor * pow(saturate(dot(h, n)), g_Shine);
	color *= g_LightDiffuse;
	color *= shadow;
	color += g_MaterialAmbientColor * TexColor;
	return color;
}


//--------------------------------------------------------------------------------------
// Flat pixel shader
//--------------------------------------------------------------------------------------
float4 CubeMap_PS(CUBEMAP_VS_OUTPUT In) : COLOR0
{
	float4 Output;
	
	// view and light directions
	float3 v = normalize(g_Eye - In.WorldPos);
	float3 l = normalize(g_WorldLightDir);
	
	float3 n = normalize(In.Normal);
	
	// Lookup texture
	float3 TexColor = tex2D(ColorTextureSampler, In.TextureUV);
	
	Output.rgb = BlinnPhong(TexColor, n, v, l);
	Output.a = 1.0;
	
	if (g_reflections)
	{
		// Lookup cube map
		float3 reflect_color = texCUBE(CubeMapSampler, In.Reflect);
		
		// Combine with previous color
		Output.rgb = lerp(Output.rgb, reflect_color, g_reflectance);
	}

	return Output;
}


//--------------------------------------------------------------------------------------
// This function computes barycentric coordinates
//--------------------------------------------------------------------------------------
float3 Barycentric(in float3 v0, in float3 v1, in float3 v2, in float3 q)
{
	float3 e1 = v1 - v0;
	float3 e2 = v2 - v0;
	float3 f = q - v0;
	float e1_dot_e1 = dot(e1, e1);
	float e1_dot_e2 = dot(e1, e2);
	float e2_dot_e2 = dot(e2, e2);
	float B = dot(e2_dot_e2 * e1 - e1_dot_e2 * e2, f);
	float C = dot(e1_dot_e1 * e2 - e1_dot_e2 * e1, f);
	float D = e1_dot_e1 * e2_dot_e2 - e1_dot_e2 * e1_dot_e2;
	float3 bary;
	bary.y = B / D;
	bary.z = C / D;
	bary.x = 1.f - bary.y - bary.z;
	return bary;
}


//--------------------------------------------------------------------------------------
// This function computes distance from p to plane along u
// p - ray origin
// u - ray direction
// v0, v1, v2 - plain pints (noncollinear)
//--------------------------------------------------------------------------------------
float RayPlaneDistance(float3 p, float3 u, float3 v0, float3 v1, float3 v2)
{
	// Plane normal
	float3 n = cross(v1 - v0, v2 - v0);
	
	// Distance from origin along normal
	float d = dot(n, v0);
	
	return (d - dot(p, n)) / dot(u, n);
}


//--------------------------------------------------------------------------------------
// This function finds first ray versus plane intersection (returns false if there is no intersection)
// base_face - face, containing p (it excluded from intersrction testing)
// p - ray origin
// u - ray direction
// face - intersection face
// q - intersection point
// bary - barycentric coordinates of q
//--------------------------------------------------------------------------------------
bool RayTrace(in int base_face, in float3 p, in float3 u, out int face, out float3 q, out float3 bary)
{
	float dist = 1e38; // distance from p to plane along u
	bool intersect = false;
	for (int cur_face = 0; cur_face < g_face_count; cur_face++)
		if (cur_face != base_face)
		{
			// Fetch 3 face vertices
			float3 v0, v1, v2;
			float4 t = { (float)cur_face / g_face_count + 0.5f / g_vertex_count, 0, 0, 0 };
			v0 = tex1Dlod(PositionSampler, t);
			t.x += 1.f / g_vertex_count;
			v1 = tex1Dlod(PositionSampler, t);
			t.x += 1.f / g_vertex_count;
			v2 = tex1Dlod(PositionSampler, t);
/*			v0 = PositionArray[cur_face + 0];
			v1 = PositionArray[cur_face + 1];
			v2 = PositionArray[cur_face + 2];*/
			
			// Find distance from p to current plane along p
			float cur_dist = RayPlaneDistance(p, u, v0, v1, v2);
			
			// Find intersection point
			float3 cur_q = p + cur_dist * u;
			
			if (cur_dist > 0 && cur_dist < dist)
			{
				// Compute barycenrtic coordinates
				float3 cur_bary = Barycentric(v0, v1, v2, cur_q);
				
				if (all(cur_bary >= 0))
				{
					dist = cur_dist;
					face = cur_face;
					q = cur_q;
					bary = cur_bary;
					intersect = true;
				}
			}
		}
	return intersect;
}


//--------------------------------------------------------------------------------------
// Raytracing pixel shader
//--------------------------------------------------------------------------------------
float4 Raytrace_PS(RAYTRACE_VS_OUTPUT In) : COLOR0
{
	float4 Output;

	// view and light directions
	float3 v = normalize(g_Eye - In.WorldPos);
	float3 l = normalize(g_WorldLightDir);
	float3 obj_l = mul((float3x3)g_mWorld, g_WorldLightDir);
	
	float3 n = normalize(In.Normal);
	
	int intersect_face;
	float3 obj_q, bary;
	
	// Lookup texture
	float3 TexColor = tex2D(ColorTextureSampler, In.TextureUV).rgb;
	
	float shadow_factor;
	
	// Trace in light direction (in object space)
	if (g_cast_shadows)
		shadow_factor = !RayTrace(In.FaceID, In.ObjPos, obj_l, intersect_face, obj_q, bary);
	else
		shadow_factor = 1;
//	return float4(bary, 1);
	
	Output.rgb = BlinnPhong(TexColor, n, v, l, shadow_factor);
	Output.a = 1.0;
	
	if (g_reflections)
	{
		float3 reflect_color;
		
		if (RayTrace(In.FaceID, In.ObjPos, In.ObjReflect, intersect_face, obj_q, bary))
		{
			// Transform intersection point in world space
			float3 world_q = mul(obj_q, (float3x3)g_mWorld);
			
			float3 v0, v1, v2;
			float2 t0, t1, t2;
			float4 t = { (float)intersect_face / g_face_count + 0.5f / g_vertex_count, 0, 0, 0 };
			v0 = tex1Dlod(PositionSampler, t);
			t0 = tex1Dlod(TexcoordSampler, t);
			t.x += 1.f / g_vertex_count;
			v1 = tex1Dlod(PositionSampler, t);
			t1 = tex1Dlod(TexcoordSampler, t);
			t.x += 1.f / g_vertex_count;
			v2 = tex1Dlod(PositionSampler, t);
			t2 = tex1Dlod(TexcoordSampler, t);
/*			v0 = PositionArray[intersect_face + 0];
			t0 = TexcoordArray[intersect_face + 0];
			v1 = PositionArray[intersect_face + 1];
			t1 = TexcoordArray[intersect_face + 1];
			v2 = PositionArray[intersect_face + 2];
			t2 = TexcoordArray[intersect_face + 2];*/
			
			// View in direction of ray tracing
			v = normalize(In.WorldReflect); // normalize for lighting calculations
			
			// Compute normal
			float3 n = cross(v1 - v0, v2 - v0); // in object space
			n = normalize(mul(n, (float3x3)g_mWorld)); // in world space
			n = faceforward(n, v, n);
			
			// Compute texture coordinates
			float2 texUV = t0 * bary[0] + t1 * bary[1] + t2 * bary[2];
			
			// Lookup texture
			float3 TexColor = tex2D(ColorTextureSampler, texUV);
			
			// Trace in light direction (in object space)
			if (g_cast_shadows)
				shadow_factor = !RayTrace(intersect_face, obj_q, obj_l, intersect_face, obj_q, bary);
			else
				shadow_factor = 1;
			
			// Perform lighting
			reflect_color = BlinnPhong(TexColor, n, -v, l, shadow_factor);
//			reflect_color = bary;
		}
		else
		{
			// Lookup cube map
			reflect_color = texCUBE(CubeMapSampler, In.WorldReflect);
		}
		// Combine with previous color
		Output.rgb = lerp(Output.rgb, reflect_color, g_reflectance);
	}

	return Output;
}
/*float4 Raytrace_PS(RAYTRACE_VS_OUTPUT In) : COLOR0
{
	float4 Output;

	float3 v0, v1, v2;
	float2 t0, t1, t2;
	float4 t = { (float)In.FaceID / g_face_count + 0.5f / g_vertex_count, 0, 0, 0 };
	v0 = tex1Dlod(PositionSampler, t);
	t0 = tex1Dlod(TexcoordSampler, t);
	t.x += 1.f / g_vertex_count;
	v1 = tex1Dlod(PositionSampler, t);
	t1 = tex1Dlod(TexcoordSampler, t);
	t.x += 1.f / g_vertex_count;
	v2 = tex1Dlod(PositionSampler, t);
	t2 = tex1Dlod(TexcoordSampler, t);
	Output.rgb = Barycentric(v0, v1, v2, In.ObjPos);
	Output.a = 1.f;
Output.rgb = (float3(t0, 0));
	return Output;
}*/


//--------------------------------------------------------------------------------------
// Techniques
//--------------------------------------------------------------------------------------
technique CubeMap
{
	pass P0
	{
		VertexShader = compile vs_2_0 CubeMap_VS();
		PixelShader  = compile ps_2_0 CubeMap_PS();
	}
}

technique LocalRayRracing
{
	pass P0
	{
		VertexShader = compile vs_3_0 Raytrace_VS();
		PixelShader  = compile ps_3_0 Raytrace_PS();
	}
}