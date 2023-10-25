// SPDX-License-Identifier: Apache-2.0

use crate::sema::ast::{Namespace, Parameter, StructDecl, StructType, Type};
use once_cell::sync::Lazy;
use solang_parser::pt;

pub static BUILTIN_STRUCTS: Lazy<[(StructDecl, StructType); 3]> = Lazy::new(|| {
    [
        (
            StructDecl {
                tags: Vec::new(),
                loc: pt::Loc::Builtin,
                contract: None,
                id: pt::Identifier {
                    name: "AccountInfo".to_string(),
                    loc: pt::Loc::Builtin,
                },
                fields: vec![
                    Parameter {
                        loc: pt::Loc::Builtin,
                        id: Some(pt::Identifier {
                            name: String::from("key"),
                            loc: pt::Loc::Builtin,
                        }),
                        ty: Type::Ref(Box::new(Type::Address(false))),
                        ty_loc: None,
                        indexed: false,
                        readonly: true,
                        infinite_size: false,
                        recursive: false,
                        annotation: None,
                    },
                    Parameter {
                        loc: pt::Loc::Builtin,
                        id: Some(pt::Identifier {
                            name: String::from("lamports"),
                            loc: pt::Loc::Builtin,
                        }),
                        ty: Type::Ref(Box::new(Type::Uint(64))),
                        ty_loc: None,
                        indexed: false,
                        readonly: false,
                        infinite_size: false,
                        recursive: false,
                        annotation: None,
                    },
                    Parameter {
                        loc: pt::Loc::Builtin,
                        id: Some(pt::Identifier {
                            name: String::from("data"),
                            loc: pt::Loc::Builtin,
                        }),
                        ty: Type::Slice(Box::new(Type::Bytes(1))),
                        ty_loc: None,
                        indexed: false,
                        readonly: false,
                        infinite_size: false,
                        recursive: false,
                        annotation: None,
                    },
                    Parameter {
                        loc: pt::Loc::Builtin,
                        id: Some(pt::Identifier {
                            name: String::from("owner"),
                            loc: pt::Loc::Builtin,
                        }),
                        ty: Type::Ref(Box::new(Type::Address(false))),
                        ty_loc: None,
                        indexed: false,
                        readonly: true,
                        infinite_size: false,
                        recursive: false,
                        annotation: None,
                    },
                    Parameter {
                        loc: pt::Loc::Builtin,
                        id: Some(pt::Identifier {
                            name: String::from("rent_epoch"),
                            loc: pt::Loc::Builtin,
                        }),
                        ty: Type::Uint(64),
                        ty_loc: None,
                        indexed: false,
                        readonly: true,
                        infinite_size: false,
                        recursive: false,
                        annotation: None,
                    },
                    Parameter {
                        loc: pt::Loc::Builtin,
                        id: Some(pt::Identifier {
                            name: String::from("is_signer"),
                            loc: pt::Loc::Builtin,
                        }),
                        ty: Type::Bool,
                        ty_loc: None,
                        indexed: false,
                        readonly: true,
                        infinite_size: false,
                        recursive: false,
                        annotation: None,
                    },
                    Parameter {
                        loc: pt::Loc::Builtin,
                        id: Some(pt::Identifier {
                            name: String::from("is_writable"),
                            loc: pt::Loc::Builtin,
                        }),
                        ty: Type::Bool,
                        ty_loc: None,
                        indexed: false,
                        readonly: true,
                        infinite_size: false,
                        recursive: false,
                        annotation: None,
                    },
                    Parameter {
                        loc: pt::Loc::Builtin,
                        id: Some(pt::Identifier {
                            name: String::from("executable"),
                            loc: pt::Loc::Builtin,
                        }),
                        ty: Type::Bool,
                        ty_loc: None,
                        indexed: false,
                        readonly: true,
                        infinite_size: false,
                        recursive: false,
                        annotation: None,
                    },
                ],
                offsets: Vec::new(),
                storage_offsets: Vec::new(),
            },
            StructType::AccountInfo,
        ),
        (
            StructDecl {
                tags: Vec::new(),
                loc: pt::Loc::Builtin,
                contract: None,
                id: pt::Identifier {
                    name: "AccountMeta".to_string(),
                    loc: pt::Loc::Builtin,
                },
                fields: vec![
                    Parameter {
                        loc: pt::Loc::Builtin,
                        id: Some(pt::Identifier {
                            name: String::from("pubkey"),
                            loc: pt::Loc::Builtin,
                        }),
                        ty: Type::Ref(Box::new(Type::Address(false))),
                        ty_loc: None,
                        indexed: false,
                        readonly: false,
                        infinite_size: false,
                        recursive: false,
                        annotation: None,
                    },
                    Parameter {
                        loc: pt::Loc::Builtin,
                        id: Some(pt::Identifier {
                            name: String::from("is_writable"),
                            loc: pt::Loc::Builtin,
                        }),
                        ty: Type::Bool,
                        ty_loc: None,
                        indexed: false,
                        readonly: false,
                        infinite_size: false,
                        recursive: false,
                        annotation: None,
                    },
                    Parameter {
                        loc: pt::Loc::Builtin,
                        id: Some(pt::Identifier {
                            name: String::from("is_signer"),
                            loc: pt::Loc::Builtin,
                        }),
                        ty: Type::Bool,
                        ty_loc: None,
                        indexed: false,
                        readonly: false,
                        infinite_size: false,
                        recursive: false,
                        annotation: None,
                    },
                ],
                offsets: Vec::new(),
                storage_offsets: Vec::new(),
            },
            StructType::AccountMeta,
        ),
        (
            StructDecl {
                tags: Vec::new(),
                id: pt::Identifier {
                    name: "ExternalFunction".to_string(),
                    loc: pt::Loc::Builtin,
                },
                loc: pt::Loc::Builtin,
                contract: None,
                fields: vec![
                    Parameter {
                        loc: pt::Loc::Builtin,
                        id: None,
                        ty: Type::FunctionSelector,
                        ty_loc: None,
                        indexed: false,
                        readonly: false,
                        infinite_size: false,
                        recursive: false,
                        annotation: None,
                    },
                    Parameter {
                        loc: pt::Loc::Builtin,
                        id: None,
                        ty: Type::Address(false),
                        ty_loc: None,
                        indexed: false,
                        readonly: false,
                        infinite_size: false,
                        recursive: false,
                        annotation: None,
                    },
                ],
                offsets: Vec::new(),
                storage_offsets: Vec::new(),
            },
            StructType::ExternalFunction,
        ),
    ]
});

impl StructType {
    pub fn definition<'a>(&'a self, ns: &'a Namespace) -> &StructDecl {
        match self {
            StructType::UserDefined(struct_no) => &ns.structs[*struct_no],
            StructType::AccountInfo => &BUILTIN_STRUCTS[0].0,
            StructType::AccountMeta => &BUILTIN_STRUCTS[1].0,
            StructType::ExternalFunction => &BUILTIN_STRUCTS[2].0,
            StructType::SolParameters => unreachable!("SolParameters is defined in a solana.c"),
        }
    }
}
